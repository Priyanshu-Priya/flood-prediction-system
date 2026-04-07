"""
Dask-based Scalability Utilities
==================================
Out-of-core raster processing for large DEMs and feature stacks.
Enables processing of India-scale datasets on a single RTX 4050 machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


def setup_dask_client(
    n_workers: int = 4,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
) -> "distributed.Client":
    """
    Initialize a local Dask distributed client.

    For an RTX 4050 laptop with ~16GB RAM:
    - 4 workers × 4GB = 16GB RAM budget
    - Each worker gets 2 threads for I/O parallelism
    """
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    client = Client(cluster)

    logger.info(
        f"Dask client started | workers={n_workers} | "
        f"threads={threads_per_worker} | mem_limit={memory_limit} | "
        f"dashboard={client.dashboard_link}"
    )

    return client


def lazy_load_raster(
    filepath: Path,
    chunk_size: int = 1024,
) -> "xarray.DataArray":
    """
    Lazy-load a large raster as a Dask-backed xarray DataArray.

    No data is read from disk until .compute() is called.
    This allows processing multi-GB rasters that don't fit in RAM.

    Args:
        filepath: Path to GeoTIFF
        chunk_size: Dask chunk size in pixels

    Returns:
        Lazy xarray DataArray backed by Dask arrays
    """
    import rioxarray

    da = rioxarray.open_rasterio(
        str(filepath),
        chunks={"x": chunk_size, "y": chunk_size},
    )

    logger.info(
        f"Lazy loaded: {filepath.name} | shape={da.shape} | "
        f"chunks={da.chunks} | dtype={da.dtype}"
    )

    return da


def parallel_raster_apply(
    input_path: Path,
    func: callable,
    output_path: Path,
    chunk_size: int = 512,
    **func_kwargs,
) -> Path:
    """
    Apply a function to a raster in parallel using Dask.

    The function is applied to each chunk independently,
    enabling embarrassingly parallel processing.

    Args:
        input_path: Input raster path
        func: Function that takes a numpy array and returns a numpy array
        output_path: Output raster path
        chunk_size: Processing chunk size
        **func_kwargs: Additional arguments for func

    Returns:
        Path to output raster
    """
    import dask.array as da
    import rasterio

    logger.info(f"Parallel raster processing: {input_path.name} → {output_path.name}")

    with rasterio.open(str(input_path)) as src:
        meta = src.meta.copy()
        data = src.read(1)

    # Convert to dask array
    dask_array = da.from_array(data, chunks=(chunk_size, chunk_size))

    # Apply function block-wise
    result = dask_array.map_blocks(func, dtype=np.float32, **func_kwargs)

    # Compute and save
    result_np = result.compute()

    meta.update({"dtype": "float32", "compress": "lzw"})
    with rasterio.open(str(output_path), "w", **meta) as dst:
        dst.write(result_np, 1)

    logger.info(f"Parallel processing complete: {output_path}")
    return output_path
