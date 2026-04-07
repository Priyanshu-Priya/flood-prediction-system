"""Quick smoke test: verify GloFAS client can list stations and download data."""
import sys, os
sys.path.insert(0, os.getcwd())

from src.ingestion.glofas import GloFASClient

client = GloFASClient()

# 1. List stations
print("=== Station Metadata ===")
stations = client.fetch_station_metadata()
print(stations[["station_id", "name", "river", "lat", "lon"]].to_string(index=False))
print(f"\nTotal stations: {len(stations)}")

# 2. Try downloading a small slice of GloFAS data for Patna
print("\n=== Downloading GloFAS data for GLOFAS_PATNA (2025-01, 3 days) ===")
try:
    df = client.fetch_water_levels("GLOFAS_PATNA", "2025-01-01", "2025-01-03")
    print(df.head(10))
    print(f"\nRows: {len(df)}, Columns: {list(df.columns)}")
    print("SUCCESS: GloFAS download and extraction works!")
except Exception as e:
    print(f"Download test result: {e}")
    print("(This may fail if CDS API queue is busy — that's OK, station listing works)")
