import multiprocessing
multiprocessing.set_start_method("fork")
import os
import torch
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.merge import merge
from datasets import SWETsunamiForPlotting
from scipy.interpolate import griddata
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#activate conda env topos

## data source ##
### https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm ###
#
# f1 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N090E_20101117_gmted_mea075.tif')
# dta1 = f1.read()
# f2 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N120E_20101117_gmted_mea075.tif')
# dta2 = f2.read()
# f3 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N150E_20101117_gmted_mea075.tif')
# dta3 = f3.read()
# f4 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S090E_20101117_gmted_mea075.tif')
# dta4 = f4.read()
# f5 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S120E_20101117_gmted_mea075.tif')
# dta5 = f5.read()
# f6 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S150E_20101117_gmted_mea075.tif')
# dta6 = f6.read()
# f7 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N090E_20101117_gmted_mea075.tif')
# dta7 = f7.read()
# f8 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N120E_20101117_gmted_mea075.tif')
# dta8 = f8.read()
# f9 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N150E_20101117_gmted_mea075.tif')
# dta9 = f9.read()
# f10 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N090E_20101117_gmted_mea075.tif')
# dta10 = f10.read()
# f11 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N120E_20101117_gmted_mea075.tif')
# dta11 = f11.read()
# f12 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N150E_20101117_gmted_mea075.tif')
# dta12 = f12.read()
# #
# combined_data = merge([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
# full_data = combined_data[0][0]
# transform = combined_data[1]
# topo_dataset = rasterio.open('/Users/emcdugald/tsunseiver/geodata/merged.tif', 'w',driver='GTiff',
#     height=full_data.shape[0],width=full_data.shape[1],count=1,
#     dtype=full_data.dtype,crs='+proj=latlong',transform=transform)
# topo_dataset.write(full_data, 1)
# topo_dataset.close()
topo = rasterio.open('/Users/emcdugald/sparse_sens_tsunami/geodata/merged.tif')
topo_data = topo.read()

import geopandas as gpd
# from shapely.geometry import mapping
# from rasterio import mask as msk
#
# def clip_raster(gdf, img):
#     clipped_array, clipped_transform = msk.mask(img, [mapping(gdf.iloc[0].geometry)])
#     clipped_array, clipped_transform = msk.mask(img, [mapping(gdf.iloc[0].geometry)], nodata=(np.amax(clipped_array[0]) + 1))
#     clipped_array[0] = clipped_array[0] + abs(np.amin(clipped_array))
#     value_range = np.amax(clipped_array) + abs(np.amin(clipped_array))
#     return clipped_array, value_range
#
df = gpd.read_file('/Users/emcdugald/sparse_sens_tsunami/geodata/ne_10m_admin_0_countries.shp')
#
chi = df.loc[df['ADMIN'] == 'China']
# clipped_array_chi, clipped_transform_chi = msk.mask(topo, [mapping(chi.iloc[0].geometry)])
# china_topography, china_value_range = clip_raster(chi, topo)
# # # # #
jap = df.loc[df['ADMIN'] == 'Japan']
# clipped_array_jap, clipped_transform_jap = msk.mask(topo, [mapping(jap.iloc[0].geometry)])
# japan_topography, japan_value_range = clip_raster(jap, topo)
# # # # #
rus = df.loc[df['ADMIN'] == 'Russia']
# clipped_array_rus, clipped_transform_rus = msk.mask(topo, [mapping(rus.iloc[0].geometry)])
# russia_topography, russia_value_range = clip_raster(rus, topo)
# # # # #
sko = df.loc[df['ADMIN'] == 'South Korea']
# clipped_array_sko, clipped_transform_sko = msk.mask(topo, [mapping(sko.iloc[0].geometry)])
# sko_topography, sko_value_range = clip_raster(sko, topo)
# # # # #
nko = df.loc[df['ADMIN'] == 'North Korea']
# clipped_array_nko, clipped_transform_nko = msk.mask(topo, [mapping(nko.iloc[0].geometry)])
# nko_topography, nko_value_range = clip_raster(nko, topo)
# # # # #
phi = df.loc[df['ADMIN'] == 'Philippines']
# clipped_array_phi, clipped_transform_phi = msk.mask(topo, [mapping(phi.iloc[0].geometry)])
# phi_topography, phi_value_range = clip_raster(phi, topo)
# # # # # #
tai = df.loc[df['ADMIN'] == 'Taiwan']
# clipped_array_tai, clipped_transform_tai = msk.mask(topo, [mapping(tai.iloc[0].geometry)])
# tai_topography, tai_value_range = clip_raster(tai, topo)
# # # # #
vie = df.loc[df['ADMIN'] == 'Vietnam']
# clipped_array_vie, clipped_transform_vie = msk.mask(topo, [mapping(vie.iloc[0].geometry)])
# vie_topography, vie_value_range = clip_raster(vie, topo)
# # # # #
mon = df.loc[df['ADMIN'] == 'Mongolia']
# clipped_array_mon, clipped_transform_mon = msk.mask(topo, [mapping(mon.iloc[0].geometry)])
# mon_topography, mon_value_range = clip_raster(mon, topo)
# mon_topography = mon_topography - np.min(mon_topography)


min_lat = -10
max_lat = 70
min_lon = 90
max_lon = 180
min_lat_plot = -8
max_lat_plot = 65
min_lon_plot = 110
max_lon_plot = 180
nlat , nlon = np.shape(topo_data[0])
lat_vals = np.linspace(max_lat,min_lat,nlat)
lon_vals = np.linspace(min_lon,max_lon,nlon)
min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx])
lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
dx = (lon_vals_new[1]-lon_vals_new[0])/2.
dy = (lat_vals_new[1]-lat_vals_new[0])/2.
extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]
# #
# topo_vals = china_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + japan_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + russia_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + sko_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + nko_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + phi_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + tai_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + vie_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]\
#           + mon_topography[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
# # topo_vals = china_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + japan_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + russia_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + sko_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + nko_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + phi_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + tai_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]\
# #           + vie_topography[0][min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]
# #
topo_vals = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
print("topo_vals_shape:",np.shape(topo_vals))
topo_vals = topo_vals[::80,::80]
lat_vals = lat_vals_new[::80]
lon_vals = lon_vals_new[::80]
print("new topo_vals_shape:",np.shape(topo_vals))


def coord_idx(s):
    if 0 <= s <= 144:
        return 0, 0, 0
    elif 145 <= s <= 289:
        return 1, 145, 0
    elif 290 <= s <= 434:
        return 2, 290, 0
    elif 435 <= s <= 579:
        return 3, 435, 0
    elif 580 <= s <= 724:
        return 4, 580, 0
    elif 725 <= s <= 869:
        return 5, 725, 0
    elif 870 <= s <= 1014:
        return 6, 870, 0
    else:
        return 7, 1015, 0

def training_lons():
    return np.array([136.6180, 139.5560, 139.3290, 138.9350, 140.9290, 135.7400, 141.5010, 142.3870])

def training_lats():
    return np.array([33.0700, 28.8560, 28.9320, 29.3840, 33.4530, 33.1570, 35.9360, 35.2670])

def unseen_lons():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])

def unseen_lats():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])

# ## 0-2 hr models here ###
#fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat"
#fname = "agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_6920_wd_tf_145.mat"
fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145_as.mat"
#fname = "agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_6920_wd_tf_145_as.mat"
#fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat"
#fname = "agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_6920_wd_tf_145.mat"
ver_num = 21
#ver_num = 34
#ver_num = 35
type = "unseen"
#type = "training"
st_time = 0


slice=900
#
# # ### USE FOR AS COMPARISON ###
min_lat_plot = 25
max_lat_plot = 43
min_lon_plot = 132
max_lon_plot = 150

# min_lat_plot = 25
# max_lat_plot = 50
# min_lon_plot = 130
# max_lon_plot = 160


# # ## 2-4 hr models here ##
# #fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
# #fname = "agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
# fname = "unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
# #fname = "agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_289.mat"
# ver_num = 25
# type = "unseen"
# #type = "training"
# st_time = 145
#
# slice=100
#
# min_lat_plot = 0
# max_lat_plot = 55
# min_lon_plot = 115
# max_lon_plot = 180



dx = (lon_vals[1]-lon_vals[0])/2.
dy = (lat_vals[1]-lat_vals[0])/2.
min_lat_idx = np.argsort(np.abs(lat_vals-min_lat_plot))[0]
max_lat_idx = np.argsort(np.abs(lat_vals-max_lat_plot))[0]
min_lon_idx = np.argsort(np.abs(lon_vals-min_lon_plot))[0]
max_lon_idx = np.argsort(np.abs(lon_vals-max_lon_plot))[0]
extent = [lon_vals[min_lon_idx]-dx, lon_vals[max_lon_idx]+dx,
          lat_vals[min_lat_idx]-dy, lat_vals[max_lat_idx]+dy]


out_path = os.getcwd()+"/lightning_logs/version_"+str(ver_num)+"/"
output_im = torch.load(out_path+'tensor'+'_'+str(type)+'.pt')

true_data, latitude, longitude, mask, max_ht, times, sensors, div = SWETsunamiForPlotting(fname)
true_data *= max_ht
output_im *= max_ht

# NEED INTERPOLATED WAVE_HEIGT, LONGITUDE, LATITUDE, MASK


if not os.path.exists(out_path+"/recons/"+type):
    if not os.path.exists(out_path + "/recons"):
        os.mkdir(out_path+"/recons")
    os.mkdir(out_path+"/recons/"+type)

if type == 'training':
    epi_lons = training_lons()
    epi_lats = training_lats()
    epi_idx, start_time, time_shift = coord_idx(slice)
else:
    epi_lons = unseen_lons()
    epi_lats = unseen_lats()
    epi_idx, start_time, time_shift = coord_idx(slice)

tsun_lons_tmp = longitude*(180 / np.pi)
tsun_lats_tmp = latitude*(180 / np.pi)
tsun_lons = tsun_lons_tmp
tsun_lats = tsun_lats_tmp
tsun_lons = tsun_lons[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot) ]
tsun_lats = tsun_lats[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]

int_Lons, int_Lats = np.meshgrid(lon_vals, lat_vals)

epi_lon = epi_lons[epi_idx]
epi_lat = epi_lats[epi_idx]
mins = round((slice + st_time - start_time) * 50 / 60, 2)
prediction = output_im[slice].cpu().detach().numpy()

true_data_s = true_data[slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
prediction = prediction[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]

unstruct_coords = np.array([tsun_lons, tsun_lats]).T
interp_true = np.asarray(
    griddata(unstruct_coords, true_data_s, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_pred = np.asarray(
    griddata(unstruct_coords, prediction, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))

interp_true = interp_true[:,:,0]
interp_pred = interp_pred[:,:,0]
interp_abs_err = np.abs(interp_true-interp_pred)

interp_true[(topo_vals != 0)] = 0.0
interp_pred[(topo_vals != 0)] = 0.0
interp_abs_err[(topo_vals != 0)] = 0.0
min_val = min([np.min(interp_true),np.min(interp_pred),np.min(interp_abs_err)])
max_val = max([np.max(interp_true),np.max(interp_pred),np.max(interp_abs_err)])


width = (extent[1]-extent[0])
height = (extent[3]-extent[2])
aspect = width/height
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(19,5))
from numpy.ma import masked_array
topo_img = masked_array(topo_vals, topo_vals == 0.0)
lons_topo_img = masked_array(int_Lons, topo_vals == 0.0)
lats_topo_img = masked_array(int_Lats, topo_vals == 0.0)
true_tsu_img = masked_array(interp_true, topo_vals != 0.0)
lons_tsu_img = masked_array(int_Lons, topo_vals != 0.0)
lats_tsu_img = masked_array(int_Lats, topo_vals != 0.0)
pred_tsu_img = masked_array(interp_pred, topo_vals != 0.0)
abs_err_img = masked_array(interp_abs_err, topo_vals != 0.0)
shw1 = axs[0].contourf(lons_tsu_img,np.flip(lats_tsu_img),true_tsu_img,levels=50,extent=extent,cmap='bwr')
shw2 = axs[0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw3 = axs[1].contourf(lons_tsu_img,np.flip(lats_tsu_img),pred_tsu_img,levels=50,extent=extent,cmap='bwr')
shw4 = axs[1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw5 = axs[2].contourf(lons_tsu_img,np.flip(lats_tsu_img),abs_err_img,levels=50,extent=extent,cmap='bwr')
shw6 = axs[2].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=50,extent=extent,cmap='summer')
shw1.set_clim(min_val,max_val)
shw3.set_clim(min_val,max_val)
shw5.set_clim(min_val,max_val)
import shapely.ops as sops
new_shape = sops.unary_union([el for el in chi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in rus['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in jap['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in nko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in sko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in phi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in tai['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in vie['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[2].plot(*geom.exterior.xy, c='k', lw=1.0)
sensor_longs = sensors[:, 0]
sensor_lats = sensors[:, 1]
axs[0].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=60, marker="^",color='yellow',edgecolor='k')
axs[0].scatter(epi_lon,epi_lat,s=120,marker="x",color='k')

##USE BELOW FOR FIRST RECON FIG
#axs[0].scatter((sensor_longs * (180 / np.pi))[28],(sensor_lats * (180 / np.pi))[28],s=100,facecolors='none', edgecolors='k',linewidth=3)
##USE BELOW FOR SECOND RECON FIG
#axs[0].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=100,facecolors='none', edgecolors='k', linewidth=3)
#axs[0].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=100,facecolors='none', edgecolors='k',linewidth=3)

axs[0].set_xlim([extent[0], extent[1]])
axs[0].set_ylim([extent[2],extent[3]])
axs[1].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=60, marker="^",color='yellow',edgecolor='k')
axs[1].scatter(epi_lon,epi_lat,s=120,marker="x",color='k')

##USE BELOW FOR FIRST RECON FIG ##
#axs[1].scatter((sensor_longs * (180 / np.pi))[28],(sensor_lats * (180 / np.pi))[28],s=100,facecolors='none', edgecolors='k',linewidth=3)
##USE BELOW FOR SECOND RECON FIG ##
#axs[1].scatter((sensor_longs * (180 / np.pi))[14],(sensor_lats * (180 / np.pi))[14],s=100,facecolors='none', edgecolors='k',linewidth=3)
#axs[1].scatter((sensor_longs * (180 / np.pi))[23],(sensor_lats * (180 / np.pi))[23],s=100,facecolors='none', edgecolors='k',linewidth=3)

axs[1].set_xlim([extent[0], extent[1]])
axs[1].set_ylim([extent[2],extent[3]])
#axs[2].scatter(sensor_longs * (180 / np.pi), sensor_lats * (180 / np.pi), s=20, marker="^",color='m')
axs[2].set_xlim([extent[0], extent[1]])
axs[2].set_ylim([extent[2],extent[3]])
axs[0].set_title('True ($m$)')
axs[1].set_title('Predicted')
axs[2].set_title('Absolute Error')
axs[0].set_aspect(aspect)
axs[1].set_aspect(aspect)
axs[2].set_aspect(aspect)
axs[0].set_ylabel(r'Latitude ($^\circ$N)', labelpad=10)
axs[0].set_xlabel(r'Longitude ($^\circ$E)', labelpad=10)

fig.suptitle(r"Epicenter: ({}$^\circ$E, {}$^\circ$N), Time: {} mins".format(epi_lon, epi_lat, mins),y=1.0)
fig.colorbar(shw1,ax=axs.ravel().tolist(),fraction=0.047, pad=0.04)
axs[0].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
axs[1].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
axs[2].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
#plt.savefig("/Users/emcdugald/sparse_sens_tsunami/lightning_logs/version_{}/recons/{}/".format(ver_num,type)+"interp_true_vs_pred_" + str(fname) + "_slice" + str(slice) + "_v4.pdf",dpi=400,bbox_inches='tight')
plt.savefig("/Users/emcdugald/sparse_sens_tsunami/for_paper/version_{}/recons/{}/".format(ver_num,type)+"interp_true_vs_pred_" + str(fname) + "_slice" + str(slice) + "_1028.pdf",dpi=400,bbox_inches='tight')


### DIVERGENCE PLOTS HERE ###
div_curr = div[slice][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
div_prev = div[slice-1][(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]

interp_div_curr = np.asarray(
    griddata(unstruct_coords, div_curr, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_div_prev = np.asarray(
    griddata(unstruct_coords, div_prev, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_true_div_avg = .5*(interp_div_curr+interp_div_prev)
pred_curr = output_im[slice].cpu().detach().numpy()
pred_prev = output_im[slice-1].cpu().detach().numpy()
pred_curr = pred_curr[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
pred_prev = pred_prev[(tsun_lons_tmp >= min_lon_plot) &
                      (tsun_lons_tmp <= max_lon_plot) &
                      (tsun_lats_tmp >= min_lat_plot) &
                      (tsun_lats_tmp <= max_lat_plot)]
interp_pred_curr = np.asarray(
    griddata(unstruct_coords, pred_curr, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_pred_prev = np.asarray(
    griddata(unstruct_coords, pred_prev, (int_Lons, np.flip(int_Lats)), method='cubic', fill_value=0))
interp_pred_dhdt = (interp_pred_curr-interp_pred_prev)/50

interp_true_div_avg = interp_true_div_avg[:,:,0]
interp_pred_dhdt = interp_pred_dhdt[:,:,0]
interp_true_div_avg[(topo_vals != 0)] = 0.0
interp_pred_dhdt[(topo_vals != 0)] = 0.0
min_val = min([np.min(interp_true_div_avg),np.min(interp_pred_dhdt)])
max_val = max([np.max(interp_true_div_avg),np.max(interp_pred_dhdt)])
fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
true_div_img = masked_array(interp_true_div_avg, topo_vals != 0.0)
pred_div_img = masked_array(interp_pred_dhdt, topo_vals != 0.0)
shw1 = axs[0].contourf(lons_tsu_img,np.flip(lats_tsu_img),true_div_img,levels=120,extent=extent,cmap='bwr')
shw2 = axs[0].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=120,extent=extent,cmap='summer')
shw3 = axs[1].contourf(lons_tsu_img,np.flip(lats_tsu_img),pred_div_img,levels=120,extent=extent,cmap='bwr')
shw4 = axs[1].contourf(lons_topo_img,np.flip(lats_topo_img),topo_img,levels=120,extent=extent,cmap='summer')
shw1.set_clim(min_val,max_val)
shw3.set_clim(min_val,max_val)
new_shape = sops.unary_union([el for el in chi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in rus['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in jap['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in nko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in sko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in phi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in tai['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in vie['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    axs[0].plot(*geom.exterior.xy, c='k', lw=1.0)
    axs[1].plot(*geom.exterior.xy, c='k', lw=1.0)
axs[0].set_xlim([extent[0], extent[1]])
axs[0].set_ylim([extent[2],extent[3]])
axs[1].set_xlim([extent[0], extent[1]])
axs[1].set_ylim([extent[2],extent[3]])
axs[0].set_title('True ($m/s$)')
axs[1].set_title('Predicted')
axs[0].set_aspect(aspect)
axs[1].set_aspect(aspect)
axs[0].set_ylabel('Latitude ($^\circ$N)', labelpad=10)
axs[0].set_xlabel('Longitude ($^\circ$E)', labelpad=10)
fig.suptitle("Epicenter: ({}$^\circ$E, {}$^\circ$N), Time: {} mins".format(epi_lon, epi_lat, mins),y=1.0)
fig.colorbar(shw1,ax=axs.ravel().tolist(),fraction=.047, pad=0.04)
axs[0].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
axs[1].grid(color = 'gray', linestyle = '--', linewidth = 0.5)
#plt.savefig("/Users/emcdugald/sparse_sens_tsunami/lightning_logs/version_{}/recons/{}/".format(ver_num,type)+"div_interp_true_vs_pred_" + str(fname) + "_slice" + str(slice) + "_v4.pdf",dpi=400,bbox_inches='tight')
plt.savefig("/Users/emcdugald/sparse_sens_tsunami/for_paper/version_{}/recons/{}/".format(ver_num,type)+"div_interp_true_vs_pred_" + str(fname) + "_slice" + str(slice) + "_1028.pdf",dpi=400,bbox_inches='tight')
