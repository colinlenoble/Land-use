import os
os.environ['ESMFMKFILE'] = "/gpfs/workdir/shared/juicce/envs/xenv/lib/esmf.mk"
import xesmf as xe
import xarray as xr
import xagg as xg
import dask.array as da
import numpy as np
import pandas as pd
import glob
import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
import rasterio
import rioxarray as rxr
from rasterio.warp import reproject, Resampling, calculate_default_transform
from joblib import Parallel, delayed
from scipy.stats import fisk, norm

def load_shapefile(shapefile_path, france=False):
    nuts3 = gpd.read_file(shapefile_path)
    nuts3['geometry'] = nuts3.buffer(1000)
    nuts3 = nuts3.to_crs("EPSG:4326")
    if france:
        nuts3 = nuts3[nuts3['CNTRY'] == 'FR']
    nuts3 = nuts3[['LAU_ID', 'geometry', 'LAU_NAME', 'POP_2023', 'AREA_KM2', 'NUTS3_ID']]
    nuts3 = nuts3.to_crs("EPSG:4326")
    return nuts3


def agg_climate_raster(raster_path):
    df_meta = pd.read_csv('/gpfs/workdir/shared/juicce/land_use/data/SAFRAN/coordonnees_grille_safran_lambert-2-etendu.csv', sep=';', encoding='latin1')
    df_meta['ID'] = df_meta['LAMBX (hm)'].astype(str) + '_' + df_meta['LAMBY (hm)'].astype(str)
    df_meta.drop(['LAMBX (hm)', 'LAMBY (hm)'], axis=1, inplace=True)
    df_meta = df_meta.set_index(['ID'], drop=True)

    colnames = ['LAMBX', 'LAMBY', 'DATE', 'PRELIQ_Q', 'T_Q', 'TSUP_H_Q', 'ETP_Q', 'SWI_Q']
    df = pd.read_csv(raster_path, compression='gzip', sep=';', header=0)
    df = df[colnames]
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['ID'] = df['LAMBX'].astype(str) + '_' + df['LAMBY'].astype(str)
    df.drop(columns=['LAMBX', 'LAMBY'], inplace=True)
    df = df.rename(columns={'PRELIQ_Q': 'precipitation', 'T_Q': 'temperature',
                                'TSUP_H_Q': 'temperature_max', 'ETP_Q': 'evapotranspiration', 'SWI_Q': 'soil_moisture', 'DATE': 'time'})
    df['big_id'] = df['ID'].astype(str) + '_' + df['time'].dt.strftime('%Y%m%d') 
    df.set_index(['big_id'], inplace=True)
    start_date = raster_path.split('/')[-1].split('_')[2].split('-')[0]
    end_date = raster_path.split('/')[-1].split('_')[2].split('-')[1].split('.')[0]
    df = df.to_xarray()
    df.to_netcdf(path=raster_path.replace('.csv.gz', '.nc'), format='NETCDF4', engine='netcdf4')

def closest_id(shapefile):
    shapefile['centroid_x'] = shapefile.geometry.centroid.x
    shapefile['centroid_y'] = shapefile.geometry.centroid.y

    shapefile['centroid_x'] = shapefile['centroid_x'].astype(float)
    shapefile['centroid_y'] = shapefile['centroid_y'].astype(float)

    df_meta = pd.read_csv('/gpfs/workdir/shared/juicce/land_use/data/SAFRAN/coordonnees_grille_safran_lambert-2-etendu.csv', sep=';', encoding='latin1')
    df_meta['ID'] = df_meta['LAMBX (hm)'].astype(str) + '_' + df_meta['LAMBY (hm)'].astype(str)
    df_meta.drop(['LAMBX (hm)', 'LAMBY (hm)'], axis=1, inplace=True)
    df_meta = df_meta.set_index(['ID'], drop=True)

    #find the closest pixel in df for each lau_2023 in df_meta to keep it
    df_centroid = df_meta[['LAT_DG', 'LON_DG']].reset_index()
    df_centroid['LAT_DG'] = df_centroid['LAT_DG'].str.replace(',', '.').astype(float)
    df_centroid['LON_DG'] = df_centroid['LON_DG'].str.replace(',', '.').astype(float)

    df_closest = shapefile[['LAU_ID', 'centroid_x', 'centroid_y']].copy()
    for index, row in shapefile.iterrows():
        # Calculate the distance between the centroid and each pixel in df_meta
        df_centroid['distance'] = np.sqrt((df_centroid['LAT_DG'] - row['centroid_y'])**2 + (df_centroid['LON_DG'] - row['centroid_x'])**2)
        # Find the index of the closest pixel
        closest_index = df_centroid['distance'].idxmin()
        # Get the ID of the closest pixel
        closest_id = df_centroid.loc[closest_index, 'ID']
        
        # Add the ID to the df_closest DataFrame
        df_closest.at[index, 'closest_id'] = closest_id
        df_closest.at[index, 'closest_distance'] = df_centroid.loc[closest_index, 'distance']
    return df_closest[['LAU_ID', 'closest_id']]


def parameters_loglogistic(dataframe, column):
    x = dataframe[column].dropna()
    params = fisk.fit(x)
    c, loc, scale = params
    return c, loc, scale

def standardize_loglogistic(c, loc, scale, dataframe, column):
    cdf_values = fisk.cdf(dataframe[column], c, loc=loc, scale=scale)
    cdf_values = np.clip(cdf_values, 1e-6, 1 - 1e-6)
    standardized = norm.ppf(cdf_values)
    dataframe[f"{column}_standardized"] = standardized
    return dataframe



def make_spei(paths_list, shapefile_path, spei_path):
    # Load the shapefile
    # nuts3 = load_shapefile(shapefile_path, france=True)
    # # Get the closest id for each LAU_ID
    # df_closest = closest_id(nuts3)
    # Load the SAFRAN data
    ds = xr.open_mfdataset(paths_list, combine='by_coords', parallel=True)
    ds['ID'] = ds['big_id'].str.split('_').str[0]
    ds['time'] = pd.to_datetime(ds['big_id'].str.split('_').str[1], format='%Y%m%d')
    #put as a dimension the time
    ds = ds.set_index({'big_id': 'ID', 'time': 'time'})
    # Ensure the time dimension is monotonic by sorting
    ds = ds.sortby('time')
    # Ensure the time dimension is monotonic by sorting

    # Reproject the data to the shapefile CRS
    ds['def_pr'] = ds['prec'] - ds['evapotranspiration']
    ds['def_pr'] = ds['def_pr'].rolling(time=30, center=True).mean()
    ds = ds['def_pr']
    # Initialize lists to store parameters
    def compute_parameters(group):
        # Calculate the log-logistic parameters for the 'def_pr' column
        c, loc, scale = parameters_loglogistic(group, 'def_pr')
        return pd.Series({'c': c, 'loc': loc, 'scale': scale})

    # Group the dataset by 'ID' and compute parameters for each group
    parameters_df = ds.sel(time=slice('1970-01-01', '1999-12-31')).to_dataframe().groupby('ID').apply(compute_parameters).reset_index()

    # Extract the parameters into separate lists
    c_list = parameters_df['c'].tolist()
    loc_list = parameters_df['loc'].tolist()
    scale_list = parameters_df['scale'].tolist()

    #apply the standardization to the dataset
    ds = ds.to_dataframe().reset_index()
    def standardize_group(group, c_list, loc_list, scale_list):
        # Standardize the 'def_pr' column using the log-logistic parameters
        group = standardize_loglogistic(c_list, loc_list, scale_list, group, 'def_pr')
        return group
    # Group the dataset by 'ID' and standardize each group

    ds = ds.groupby('ID').apply(standardize_group, c_list, loc_list, scale_list).reset_index()
    ds['year'] = ds['time'].dt.year
    ds['month'] = ds['time'].dt.month
    ds = ds.groupby(['ID', 'year', 'month']).mean().reset_index()
    ds.to_csv(spei_path, index=False)

def anomaly_temperature(paths_list, temp_anomaly_path):
    # Load the SAFRAN data
    ds = xr.open_mfdataset(paths_list, combine='by_coords', parallel=True)

    #from big_id the unique dimension break it to ID and time
    ds['ID'] = ds['big_id'].str.split('_').str[0]
    ds['time'] = pd.to_datetime(ds['big_id'].str.split('_').str[1], format='%Y%m%d')
    #put as a dimension the time
    ds = ds.set_index({'big_id': 'ID', 'time': 'time'})
    #drop the big_id column
    ds = ds.drop('big_id', axis=1)
    # Ensure the time dimension is monotonic by sorting
    ds = ds.sortby('time')
    # Reproject the data to the shapefile CRS
    mean = ds.temperature.sel(time=slice('1970-01-01', '1999-12-31')).mean(dim='time').to_dataframe().reset_index()
    std = ds.temperature.sel(time=slice('1970-01-01', '1999-12-31')).std(dim='time').to_dataframe().reset_index()
    ds = ds[['temperature']].to_dataframe().reset_index()

    ds['month'] = ds['time'].dt.month
    ds['mean'] = ds.merge(mean, on=['ID', 'month'], how='left')['temperature']
    ds['std'] = ds.merge(std, on=['ID', 'month'], how='left')['temperature']
    ds['temperature_anomaly'] = (ds['temperature'] - ds['mean']) / ds['std']

    ds['positive_anomaly'] = ds['temperature_anomaly'] > norm.ppf(0.90)
    ds['negative_anomaly'] = ds['temperature_anomaly'] < norm.ppf(0.10)
    ds['year'] = ds['time'].dt.year

    ds = ds[['ID', 'year', 'month', 'positive_anomaly', 'negative_anomaly']].groupby(['ID', 'year', 'month']).sum().reset_index()
    ds.to_csv(temp_anomaly_path, index=False)

if __name__ == "__main__":
    shapefile_path = "/gpfs/workdir/shared/juicce/land_use/shp_LAU/LAU_2013.shp"
    # nuts3 = load_shapefile(shapefile_path)
    path_list = ['QUOT_SIM2_previous-2020-202503.csv.gz',
            'QUOT_SIM2_2010-2019.csv.gz',
            'QUOT_SIM2_2000-2009.csv.gz',
            'QUOT_SIM2_1990-1999.csv.gz',
            'QUOT_SIM2_1980-1989.csv.gz',
            'QUOT_SIM2_1970-1979.csv.gz',
            'QUOT_SIM2_latest-20250301-20250426.csv.gz']
    path_list = ["/gpfs/workdir/shared/juicce/land_use/data/SAFRAN/" + path for path in path_list]

    paths_list = [path.replace('.csv.gz', '.nc') for path in path_list]
    for path in paths_list:
        agg_climate_raster(path.replace('.nc', '.csv.gz'))
    make_spei(paths_list, shapefile_path, '/gpfs/workdir/shared/juicce/land_use/data/SAFRAN/spei.csv')
    anomaly_temperature(paths_list, '/gpfs/workdir/shared/juicce/land_use/data/SAFRAN/temperature_anomaly.csv')

            
