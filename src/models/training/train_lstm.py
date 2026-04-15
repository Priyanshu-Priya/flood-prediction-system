"""
LSTM Training Pipeline with Optuna HPO
========================================
Complete training loop for the FloodLSTM model:
- Walk-forward temporal cross-validation (no data leakage)
- Mixed-precision training (FP16 on RTX 4050)
- Gradient clipping, early stopping, cosine LR scheduling
- Optuna Bayesian hyperparameter optimization
- Checkpoint saving with best model tracking

Walk-Forward CV avoids temporal leakage:
  Fold 1: Train [2015-2018] → Validate [2019]
  Fold 2: Train [2015-2019] → Validate [2020]
  Fold 3: Train [2015-2020] → Validate [2021]
  ...
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config.settings import settings, MODEL_DIR
from src.evaluation.metrics import HydrologicalMetrics
from src.models.lstm_forecaster import FloodDataset, FloodLSTM, GaussianNLLLoss
from src.utils.metrics_logger import log_metrics


class LSTMTrainer:
    """
    Trains the FloodLSTM model with production-grade training loop.

    Features:
    - Mixed-precision training (AMP) for 2× speedup on RTX 4050
    - Gradient clipping to prevent exploding gradients in LSTMs
    - Cosine annealing LR schedule with warm restarts
    - Early stopping with patience tracking
    - Best model checkpoint saving
    - Comprehensive metric logging per epoch
    """

    def __init__(
        self,
        n_dynamic_features: int = 12,
        n_static_features: int = 8,
        n_forecast_met_features: int = 4,
        device: Optional[str] = None,
    ):
        self.device = device or settings.lstm.device
        self.n_dynamic = n_dynamic_features
        self.n_static = n_static_features
        self.n_forecast_met = n_forecast_met_features

        # Validate CUDA
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Training on GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    def train(
        self,
        train_dataset: FloodDataset,
        val_dataset: FloodDataset,
        model_params: Optional[dict] = None,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_dataset: Training FloodDataset
            val_dataset: Validation FloodDataset
            model_params: Override model hyperparameters
            output_dir: Where to save checkpoints

        Returns:
            Training results dictionary with best metrics
        """
        if output_dir is None:
            output_dir = MODEL_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build model
        params = self._get_model_params(model_params)
        model = FloodLSTM(**params).to(self.device)

        # Loss, optimizer, scheduler
        criterion = GaussianNLLLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.lstm.learning_rate,
            weight_decay=settings.lstm.weight_decay,
        )

        if settings.lstm.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.lstm.batch_size,
            shuffle=True,
            num_workers=settings.lstm.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=settings.lstm.batch_size,
            shuffle=False,
            num_workers=settings.lstm.num_workers,
            pin_memory=True,
        )

        # Mixed precision scaler
        scaler = GradScaler(enabled=settings.lstm.mixed_precision and self.device == "cuda")

        # Training state
        best_val_loss = float("inf")
        best_val_nse = -float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_nse": [], "lr": []}

        logger.info(
            f"Starting training | epochs={settings.lstm.max_epochs} | "
            f"batch_size={settings.lstm.batch_size} | device={self.device}"
        )

        for epoch in range(1, settings.lstm.max_epochs + 1):
            epoch_start = time.time()

            # ── Train ──
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )

            # ── Validate ──
            val_loss, val_predictions, val_targets = self._validate_epoch(
                model, val_loader, criterion
            )

            # Compute NSE
            val_nse = HydrologicalMetrics.nash_sutcliffe(
                val_targets, val_predictions
            )

            # Learning rate step
            current_lr = optimizer.param_groups[0]["lr"]
            if settings.lstm.scheduler == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)

            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_nse"].append(val_nse)
            history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Logging
            logger.info(
                f"Epoch {epoch:3d}/{settings.lstm.max_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"NSE={val_nse:.4f} | lr={current_lr:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            # ── Checkpointing ──
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_nse": val_nse,
                    **params,
                }
                torch.save(checkpoint, output_dir / "best_lstm_model.pt")
                logger.info(f"  ✓ New best model saved (NSE={val_nse:.4f})")
            else:
                patience_counter += 1

            # ── Early Stopping ──
            if patience_counter >= settings.lstm.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {settings.lstm.patience} epochs)"
                )
                break

        # Final summary
        logger.info(
            f"\nTraining complete!\n"
            f"  Best NSE: {best_val_nse:.4f}\n"
            f"  Best val loss: {best_val_loss:.4f}\n"
            f"  Total epochs: {epoch}\n"
            f"  Model saved: {output_dir / 'best_lstm_model.pt'}"
        )

        # Log metrics for dashboard
        log_metrics("lstm", {
            "nse_mean": round(float(best_val_nse), 4),
            "best_loss": round(float(best_val_loss), 4),
            "total_epochs": epoch,
            "data_source": "GloFAS 2019 (Seeded)"
        })

        return {
            "best_val_nse": best_val_nse,
            "best_val_loss": best_val_loss,
            "total_epochs": epoch,
            "history": history,
        }

    def _train_epoch(
        self,
        model: FloodLSTM,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
    ) -> float:
        """Single training epoch with mixed-precision support."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            history = batch["history"].to(self.device)
            static = batch["static"].to(self.device)
            target = batch["target"].to(self.device).unsqueeze(-1)
            forecast_met = batch.get("forecast_met")
            if forecast_met is not None:
                forecast_met = forecast_met.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=settings.lstm.mixed_precision and self.device == "cuda"):
                output = model(history, static, forecast_met)
                loss = criterion(output["mean"], output["std"], target)

            scaler.scale(loss).backward()

            # Gradient clipping (essential for LSTMs)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.lstm.gradient_clip
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(
        self,
        model: FloodLSTM,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Validation epoch — returns loss, predictions, and targets."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            history = batch["history"].to(self.device)
            static = batch["static"].to(self.device)
            target = batch["target"].to(self.device).unsqueeze(-1)
            forecast_met = batch.get("forecast_met")
            if forecast_met is not None:
                forecast_met = forecast_met.to(self.device)

            output = model(history, static, forecast_met)
            loss = criterion(output["mean"], output["std"], target)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(output["mean"].cpu().numpy().flatten())
            all_targets.append(target.cpu().numpy().flatten())

        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        return total_loss / max(n_batches, 1), predictions, targets

    def walk_forward_cv(
        self,
        dynamic_features: np.ndarray,
        static_features: np.ndarray,
        targets: np.ndarray,
        n_folds: int = 5,
        min_train_years: int = 3,
    ) -> dict:
        """
        Walk-forward temporal cross-validation.

        Unlike random k-fold, walk-forward preserves temporal order:
        - Training always uses past data
        - Validation always uses future data
        - No temporal leakage possible

        This gives honest out-of-sample performance estimates
        for time-series forecasting.
        """
        total_len = len(targets)
        fold_size = total_len // (n_folds + min_train_years)

        results = []
        logger.info(f"Walk-forward CV | {n_folds} folds | total_len={total_len}")

        for fold in range(n_folds):
            train_end = (min_train_years + fold) * fold_size
            val_end = min(train_end + fold_size, total_len)

            if val_end >= total_len - settings.lstm.forecast_steps:
                break

            train_dataset = FloodDataset(
                dynamic_features[:train_end],
                static_features,
                targets[:train_end],
                lookback=settings.lstm.lookback_steps,
                forecast_horizon=settings.lstm.forecast_steps,
            )

            val_dataset = FloodDataset(
                dynamic_features[train_end:val_end],
                static_features,
                targets[train_end:val_end],
                lookback=settings.lstm.lookback_steps,
                forecast_horizon=settings.lstm.forecast_steps,
            )

            logger.info(
                f"Fold {fold + 1}/{n_folds} | "
                f"train=[0:{train_end}] | val=[{train_end}:{val_end}]"
            )

            fold_results = self.train(
                train_dataset, val_dataset,
                output_dir=MODEL_DIR / f"cv_fold_{fold + 1}",
            )

            results.append({
                "fold": fold + 1,
                "best_nse": fold_results["best_val_nse"],
                "best_loss": fold_results["best_val_loss"],
                "epochs": fold_results["total_epochs"],
            })

        results_summary = {
            "folds": results,
            "mean_nse": np.mean([r["best_nse"] for r in results]),
            "std_nse": np.std([r["best_nse"] for r in results]),
        }

        logger.info(
            f"Walk-forward CV complete | "
            f"Mean NSE: {results_summary['mean_nse']:.4f} ± "
            f"{results_summary['std_nse']:.4f}"
        )

        return results_summary

    def _get_model_params(self, overrides: Optional[dict] = None) -> dict:
        """Build model params from settings with optional overrides."""
        params = {
            "n_dynamic_features": self.n_dynamic,
            "n_static_features": self.n_static,
            "n_forecast_met_features": self.n_forecast_met,
            "d_model": settings.lstm.d_model,
            "n_layers": settings.lstm.n_layers,
            "n_attention_heads": settings.lstm.n_attention_heads,
            "forecast_horizon": settings.lstm.forecast_steps,
            "dropout": settings.lstm.dropout,
        }
        if overrides:
            params.update(overrides)
        return params

    def optimize_hyperparams(
        self,
        train_dataset: FloodDataset,
        val_dataset: FloodDataset,
        n_trials: int = 30,
    ) -> dict:
        """
        Bayesian hyperparameter optimization via Optuna.

        Tunes: d_model, n_layers, learning_rate, dropout, batch_size
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.4),
                "n_attention_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            }

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

            model = FloodLSTM(
                n_dynamic_features=self.n_dynamic,
                n_static_features=self.n_static,
                **params,
            ).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = GaussianNLLLoss()

            loader = DataLoader(
                train_dataset,
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                shuffle=True,
                num_workers=2,
                drop_last=True,
            )

            # Quick train for HPO (fewer epochs)
            for epoch in range(20):
                model.train()
                for batch in loader:
                    history = batch["history"].to(self.device)
                    static = batch["static"].to(self.device)
                    target = batch["target"].to(self.device).unsqueeze(-1)

                    optimizer.zero_grad()
                    output = model(history, static)
                    loss = criterion(output["mean"], output["std"], target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Evaluate
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            _, preds, targets = self._validate_epoch(model, val_loader, criterion)
            nse = HydrologicalMetrics.nash_sutcliffe(targets, preds)

            return nse

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(
            f"HPO complete | best NSE={study.best_value:.4f} | "
            f"params={study.best_params}"
        )

        return study.best_params
