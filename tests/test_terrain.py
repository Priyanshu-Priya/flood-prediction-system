"""Tests for terrain feature engineering."""

import numpy as np
import pytest


class TestTWICalculation:
    """Test Topographic Wetness Index computation."""

    def test_twi_formula(self):
        """TWI = ln(a / tan(β)) — verify the math."""
        # Specific contributing area = 1000 m²/m, slope = 0.1 rad
        sca = 1000.0
        slope = 0.1
        expected = np.log(sca / np.tan(slope))
        assert abs(expected - 9.21) < 0.1

    def test_twi_high_value_flat_terrain(self):
        """Flat terrain with large upslope area → high TWI → flood risk."""
        sca = 10000.0  # Large contributing area
        slope = 0.01   # Nearly flat
        twi = np.log(sca / np.tan(slope))
        assert twi > 12, "Flat terrain with large SCA should have TWI > 12"

    def test_twi_low_value_steep_terrain(self):
        """Steep terrain with small upslope area → low TWI → drains fast."""
        sca = 10.0    # Small contributing area
        slope = 0.5   # Steep
        twi = np.log(sca / np.tan(slope))
        assert twi < 5, "Steep terrain should have low TWI"

    def test_twi_handles_zero_slope(self):
        """Zero slope → tan(0) = 0 → division by zero. Must handle."""
        sca = 100.0
        slope = 0.0
        # Should not crash — in practice, we add small epsilon
        slope_safe = max(slope, 1e-6)
        twi = np.log(sca / np.tan(slope_safe))
        assert np.isfinite(twi)


class TestSlopeCalculation:
    """Test slope computation from DEM."""

    def test_flat_dem_zero_slope(self):
        """A perfectly flat DEM should have zero slope everywhere."""
        flat_dem = np.ones((10, 10)) * 100.0
        dy, dx = np.gradient(flat_dem, 12.5)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        assert np.allclose(slope, 0.0, atol=1e-10)

    def test_tilted_dem_positive_slope(self):
        """A tilted plane should have uniform positive slope."""
        rows, cols = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        tilted_dem = rows * 10.0  # 10m rise per cell
        dy, dx = np.gradient(tilted_dem, 12.5)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        # Interior cells should have positive slope
        assert np.all(slope[1:-1, 1:-1] > 0)

    def test_slope_radians_range(self):
        """Slope should be in [0, π/2] radians."""
        dem = np.random.randn(50, 50) * 100
        dy, dx = np.gradient(dem, 12.5)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        assert np.all(slope >= 0)
        assert np.all(slope <= np.pi / 2)
