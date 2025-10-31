"""Download NCEP reanalysis wind data and calculate streamfunction.

This module downloads NCEP/NCAR Reanalysis 2 wind data from NOAA PSL via OPeNDAP
and calculates the atmospheric streamfunction using windspharm. Data is retrieved for
the South American summer rainy season (November-February) at 850 hPa globally,
then combined and subset to southern South America.

See https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html for data details and

> Doss-Gollin, J., Muñoz, Á. G., Mason, S. J., & Pastén, M. (2018). Heavy rainfall in Paraguay during the 2015-2016 austral summer: causes and sub-seasonal-to-seasonal predictive skill. Journal of Climate, 31(17), 6669–6685. https://doi.org/10.1175/jcli-d-17-0805.1

for details on the application of streamfunction in this region.
"""

import os
import xarray as xr
from windspharm.xarray import VectorWind
from tqdm import tqdm

# Configuration constants
START_YEAR = 1979  # NCEP Reanalysis 2 starts in 1979
END_YEAR = 2025  # Inclusive
LEVEL = 850  # Pressure level in hPa
BASE_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis2/pressure"

# Regional subset bounds (15°S-30°S, 65°W-45°W)
# NCEP uses longitude 0-360, so convert: 65°W = 295°E, 45°W = 315°E
LAT_SLICE = slice(-15, -30)
LON_SLICE = slice(295, 315)


def download_data(year, force_redownload=False):
    """Download global NCEP reanalysis wind data for a single year.

    Downloads both u and v wind components for summer months (Nov-Feb) at 850 hPa.
    Data is kept global (not regionally subset) to allow streamfunction calculation.

    Parameters
    ----------
    year : int
        Year of data to download
    force_redownload : bool, optional
        If True, download even if file exists. If False (default), skip existing files.

    Returns
    -------
    str
        Path to the output file
    """
    output_file = os.path.join("data", f"ncep_wind_global_{year}.nc")

    # Check if file exists and skip if not forcing redownload
    if os.path.isfile(output_file) and not force_redownload:
        return output_file

    # Summer months filter
    summer_months = [11, 12, 1, 2]

    # Download u and v components
    u_data_list = []
    v_data_list = []

    for variable, data_list in [("uwnd", u_data_list), ("vwnd", v_data_list)]:
        url = f"{BASE_URL}/{variable}.{year}.nc"

        # Open dataset and select pressure level
        ds = xr.open_dataset(url, decode_cf=False).sel(level=LEVEL)

        # Handle CF conventions for NCEP data
        ds[variable].attrs.pop("missing_value", None)
        ds = xr.decode_cf(ds, mask_and_scale=True, decode_times=True)

        # Filter to summer months
        data_summer = ds[variable].sel(time=ds["time"].dt.month.isin(summer_months))
        data_list.append(data_summer)

    # Combine into single dataset
    ds = xr.Dataset({"u": u_data_list[0], "v": v_data_list[0]})

    # Save combined file
    if os.path.isfile(output_file):
        os.remove(output_file)
    ds.to_netcdf(output_file, format="NETCDF4", mode="w")

    return output_file


def calculate_streamfunction(year, force_recalculate=False):
    """Calculate daily global streamfunction from wind components for one year.

    Aggregates 6-hourly wind components to daily means, then uses the windspharm
    library to compute the atmospheric streamfunction from the daily-mean winds.
    This is mathematically superior to averaging streamfunctions calculated from
    instantaneous wind fields.

    Parameters
    ----------
    year : int
        Year to process
    force_recalculate : bool, optional
        If True, recalculate even if file exists. If False (default), skip existing files.

    Returns
    -------
    str
        Path to the output file
    """
    wind_file = os.path.join("data", f"ncep_wind_global_{year}.nc")
    output_file = os.path.join("data", f"ncep_streamfunction_global_{year}.nc")

    # Check if file exists and skip if not forcing recalculation
    if os.path.isfile(output_file) and not force_recalculate:
        return output_file

    # Open the dataset
    ds = xr.open_dataset(wind_file, engine="netcdf4")

    # Filter to summer months first to avoid NaN days when resampling
    summer_months = [11, 12, 1, 2]
    ds_summer = ds.sel(time=ds.time.dt.month.isin(summer_months))

    # Resample 6-hourly winds to daily means (only for summer months)
    ds_daily = ds_summer.resample(time="1D").mean()

    # Drop any days that have NaN values (edge days between months)
    ds_daily = ds_daily.dropna(dim="time", how="any")

    # Extract u and v wind components
    uwnd = ds_daily["u"]
    vwnd = ds_daily["v"]

    # Calculate streamfunction using windspharm from daily-mean winds
    wind = VectorWind(uwnd, vwnd)
    psi = wind.streamfunction()

    # Remove output file if it exists
    if os.path.isfile(output_file):
        os.remove(output_file)

    # Save to NetCDF
    psi.to_netcdf(output_file, engine="netcdf4", format="NETCDF4", mode="w")

    # Close the input dataset
    ds.close()

    return output_file


def combine_and_subset(force_recombine=False):
    """Combine all years of streamfunction data and subset to South American region.

    Uses xarray's open_mfdataset to efficiently combine all annual streamfunction files,
    filters to summer months only (Nov, Dec, Jan, Feb), then subsets to the South American
    region and saves as a single output file.

    Parameters
    ----------
    force_recombine : bool, optional
        If True, recombine even if file exists. If False (default), skip if exists.

    Returns
    -------
    str
        Path to the output file
    """
    output_file = os.path.join(
        "data", f"ncep_streamfunction_regional_{START_YEAR}_{END_YEAR}.nc"
    )

    # Check if file exists and skip if not forcing recombine
    if os.path.isfile(output_file) and not force_recombine:
        return output_file

    # Get all streamfunction files
    pattern = os.path.join("data", "ncep_streamfunction_global_*.nc")

    # Open all files and combine
    ds = xr.open_mfdataset(pattern, combine="by_coords", engine="netcdf4")

    # Filter to summer months only (same months we downloaded wind data for)
    summer_months = [11, 12, 1, 2]
    ds_summer = ds.sel(time=ds.time.dt.month.isin(summer_months))

    # Subset to South American region
    ds_regional = ds_summer.sel(lat=LAT_SLICE, lon=LON_SLICE)

    # Remove output file if it exists
    if os.path.isfile(output_file):
        os.remove(output_file)

    # Save to NetCDF
    ds_regional.to_netcdf(output_file, engine="netcdf4", format="NETCDF4", mode="w")

    # Close the datasets
    ds.close()

    return output_file


def main(force_redownload=False):
    """Download wind data and calculate streamfunction for all years.

    Workflow:
    1. Download global wind data for each year (summer months only)
    2. Calculate global streamfunction for each year
    3. Combine all years and subset to South American region

    Parameters
    ----------
    force_redownload : bool, optional
        If True, download and recalculate even if files exist. If False (default),
        skip existing files.

    Notes
    -----
    Years to process are defined by START_YEAR and END_YEAR constants at the top of this file.
    Final output is saved as data/ncep_streamfunction_regional_{START_YEAR}_{END_YEAR}.nc
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Create year range
    years = range(START_YEAR, END_YEAR + 1)

    # Step 1: Download global wind data for each year
    print(
        f"Downloading global wind data for {len(years)} years ({START_YEAR}-{END_YEAR})..."
    )
    for year in tqdm(years, desc="Downloading wind data"):
        try:
            download_data(year, force_redownload)
        except Exception as exc:
            print(f"\nYear {year} download failed: {exc}")

    # Step 2: Calculate global streamfunction for each year
    print(
        f"Calculating streamfunction for {len(years)} years ({START_YEAR}-{END_YEAR})..."
    )
    for year in tqdm(years, desc="Calculating streamfunction"):
        try:
            calculate_streamfunction(year, force_redownload)
        except Exception as exc:
            print(f"\nYear {year} streamfunction calculation failed: {exc}")

    # Step 3: Combine all years and subset to region
    print("Combining all years and subsetting to South American region...")
    try:
        output_file = combine_and_subset(force_recombine=force_redownload)
        print(f"\nComplete! Regional streamfunction saved to {output_file}")
    except Exception as exc:
        print(f"\nCombining and subsetting failed: {exc}")


if __name__ == "__main__":
    main()
