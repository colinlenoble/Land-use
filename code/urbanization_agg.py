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




def load_shapefile(shapefile_path, france=False):
    nuts3 = gpd.read_file(shapefile_path)
    nuts3['geometry'] = nuts3.buffer(1000)
    nuts3 = nuts3.to_crs("EPSG:4326")
    if france:
        nuts3 = nuts3[nuts3['CNTRY'] == 'FR']
    return nuts3

def project_raster(raster_path, shapefile):
    #test if raster_path_projeted exists
    raster_path_projeted = raster_path.replace(".tif", "_projected.tif")
    if os.path.exists(raster_path_projeted):
        print("Projected raster already exists")
    else:
        dst_crs = "EPSG:4326"  # WGS84
        with rasterio.open(raster_path) as src:
            src_transform = src.transform

            # calculate the transform matrix for the output
            dst_transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
            )

            # set properties for output
            dst_kwargs = src.meta.copy()
            dst_kwargs.update(
                {
                    "crs": dst_crs,
                    "transform": dst_transform,
                    "width": width,
                    "height": height,
                    "nodata": 0,  # replace 0 with np.nan
                }
            )

            with rasterio.open(raster_path.replace(".tif", "_reprojected.tif"), "w", **dst_kwargs) as dst:
                # iterate through bands
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )

def agg_raster_LAU(raster_path, shapefile):
    r = rxr.open_rasterio(raster_path.replace(".tif", "_reprojected.tif"), masked=True)
    r = r.drop_vars('spatial_ref')
    r = r.reset_index('x')
    r = r.reset_index('y')

    r = r.rename({'x':'lon', 'y':'lat'})

    r_urban = r.where((r >= 10) & (r < 20))
    r_urban = r_urban.mean(dim='band')
    r_urban = r_urban.to_dataset(name='urban')
    r_urban = (r_urban>0).astype(int)
   
    r_cropland = r.where((r >= 20) & (r < 30))
    r_cropland = r_cropland.mean(dim='band')
    r_cropland = r_cropland.to_dataset(name='cropland')
    r_cropland = (r_cropland>0).astype(int)

    r_forest = r.where((r >= 30) & (r < 40))
    r_forest = r_forest.mean(dim='band')
    r_forest = r_forest.to_dataset(name='forest')
    r_forest = (r_forest>0).astype(int)

    r_wetlands = r.where((r >= 40) & (r < 50))
    r_wetlands = r_wetlands.mean(dim='band')
    r_wetlands = r_wetlands.to_dataset(name='wetlands')
    r_wetlands = (r_wetlands>0).astype(int)

    weight_map = xg.pixel_overlaps(r_urban, shapefile)
    r_urban = xg.aggregate(r_urban, weight_map).to_dataset()
    r_urban.to_netcdf(raster_path.replace(".tif", "_urban.nc"))

    r_cropland = xg.aggregate(r_cropland, weight_map).to_dataset()
    r_cropland.to_netcdf(raster_path.replace(".tif", "_cropland.nc"))

    r_forest = xg.aggregate(r_forest, weight_map).to_dataset()
    r_forest.to_netcdf(raster_path.replace(".tif", "_forest.nc"))

    r_wetlands = xg.aggregate(r_wetlands, weight_map).to_dataset()
    r_wetlands.to_netcdf(raster_path.replace(".tif", "_wetlands.nc"))



def agg_vector(gdb_path, nuts3):
    db = gpd.read_file(gdb_path, layer= gdb_path.split("/")[-1].replace(".gdb", ""))
    db_clipped = db.cx[nuts3.bounds.minx.min():nuts3.bounds.maxx.max(), nuts3.bounds.miny.min():nuts3.bounds.maxy.max()]
    db_clipped.columns = ['code'] + db_clipped.columns[1:].tolist()
    db_clipped['code'] = db_clipped['code'].astype(int)
    def process_lau_code(lau_id):
        county = nuts3[nuts3['LAU_ID'] == lau_id]
        return agg_county(db_clipped, county)

    lau_ids = nuts3['LAU_ID'].unique()
    results = Parallel(n_jobs=-1)(delayed(process_lau_code)(lau_id) for lau_id in lau_ids)

    df_final = pd.concat(results)
    return df_final


def agg_county(db, county):
    db = gpd.overlay(db, county, how='intersection')
    db['urban'] = db['code_00'].apply(lambda x: 1 if (x>=100) & (x<200) else 0)
    db['cropland'] = db['code_00'].apply(lambda x: 1 if (x>=200) & (x<300) else 0)
    db['forest'] = db['code_00'].apply(lambda x: 1 if (x>=300) & (x<400) else 0)
    db['wetland'] = db['code_00'].apply(lambda x: 1 if (x>=400) & (x<500) else 0)

    db['area'] = db.area

    db[['urban', 'cropland', 'forest', 'wetland', 'area']].sum(axis=0)

    db['urban'] = db['urban'] * db['area']
    db['cropland'] = db['cropland'] * db['area']
    db['forest'] = db['forest'] * db['area']
    db['wetland'] = db['wetland'] * db['area']

    df_final = db[['NUTS3_ID','urban', 'cropland', 'forest', 'wetland', 'area']].groupby('NUTS3_ID').sum()

    df_final['urban'] = df_final['urban'] / df_final['area']
    df_final['cropland'] = df_final['cropland'] / df_final['area']
    df_final['forest'] = df_final['forest'] / df_final['area']
    df_final['wetland'] = df_final['wetland'] / df_final['area']

    df_final = df_final.drop(columns='area')
    #add COMM_ID, LAU_ID, NUTS3_ID, LAU_name, CNTRY
    df_final = df_final.merge(nuts3[['NUTS3_ID', 'LAU_name', 'CNTRY', 'COMM_ID', 'LAU_ID']], on='NUTS3_ID', how='left')
    return df_final


if __name__ == "__main__":
    shapefile_path = "/gpfs/workdir/shared/juicce/land_use/shp_LAU/LAU_2013.shp"
    nuts3 = load_shapefile(shapefile_path)
    # raster_path_list = glob.glob("/gpfs/workdir/shared/juicce/land_use/Copernicus*.tif")
    # for raster_path in raster_path_list:
    #     project_raster(raster_path, nuts3)
    #     agg_raster_LAU(raster_path, nuts3)

    gdb_path = "/gpfs/workdir/shared/juicce/land_use/data/Copernicus_vector/*.gdb"
    for glob_path in glob.glob(gdb_path):
        df_final = agg_vector(glob_path, nuts3)
        df_final.to_csv(glob_path.replace(".gdb", ".csv"))

            
