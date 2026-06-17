import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.merge import merge

## data source ##
### https://topotools.cr.usgs.gov/gmted_viewer/viewer.htm ###

f1 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N090E_20101117_gmted_mea075.tif')
dta1 = f1.read()
f2 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N120E_20101117_gmted_mea075.tif')
dta2 = f2.read()
f3 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10N150E_20101117_gmted_mea075.tif')
dta3 = f3.read()
f4 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S090E_20101117_gmted_mea075.tif')
dta4 = f4.read()
f5 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S120E_20101117_gmted_mea075.tif')
dta5 = f5.read()
f6 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/10S150E_20101117_gmted_mea075.tif')
dta6 = f6.read()
f7 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N090E_20101117_gmted_mea075.tif')
dta7 = f7.read()
f8 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N120E_20101117_gmted_mea075.tif')
dta8 = f8.read()
f9 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/30N150E_20101117_gmted_mea075.tif')
dta9 = f9.read()
f10 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N090E_20101117_gmted_mea075.tif')
dta10 = f10.read()
f11 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N120E_20101117_gmted_mea075.tif')
dta11 = f11.read()
f12 = rasterio.open('/Users/emcdugald/tsunseiver/geodata/50N150E_20101117_gmted_mea075.tif')
dta12 = f12.read()

combined_data = merge([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12])
full_data = combined_data[0][0]
transform = combined_data[1]
topo_dataset = rasterio.open('/Users/emcdugald/tsunseiver/geodata/merged.tif', 'w',driver='GTiff',
    height=full_data.shape[0],width=full_data.shape[1],count=1,
    dtype=full_data.dtype,crs='+proj=latlong',transform=transform)
topo_dataset.write(full_data, 1)
topo_dataset.close()
topo = rasterio.open('/Users/emcdugald/sparse_sens_tsunami/geodata/merged.tif')
topo_data = topo.read()



import geopandas as gpd

df = gpd.read_file('/Users/emcdugald/sparse_sens_tsunami/geodata/ne_10m_admin_0_countries.shp')

chi = df.loc[df['ADMIN'] == 'China']
jap = df.loc[df['ADMIN'] == 'Japan']
rus = df.loc[df['ADMIN'] == 'Russia']
sko = df.loc[df['ADMIN'] == 'South Korea']
nko = df.loc[df['ADMIN'] == 'North Korea']
phi = df.loc[df['ADMIN'] == 'Philippines']
tai = df.loc[df['ADMIN'] == 'Taiwan']
vie = df.loc[df['ADMIN'] == 'Vietnam']


def training_lons():
    return np.array([136.6180, 139.5560, 139.3290, 138.9350, 140.9290, 135.7400, 141.5010, 142.3870])

def training_lats():
    return np.array([33.0700, 28.8560, 28.9320, 29.3840, 33.4530, 33.1570, 35.9360, 35.2670])

def unseen_lons_new():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])
def unseen_lats_new():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])

min_lat = -10
max_lat = 70
min_lon = 90
max_lon = 180
min_lat_plot = 25
max_lat_plot = 55
min_lon_plot = 125
max_lon_plot = 155
nlat , nlon = np.shape(topo_data[0])
lat_vals = np.linspace(max_lat,min_lat,nlat)
lon_vals = np.linspace(min_lon,max_lon,nlon)
min_lat_idx = np.where(lat_vals >= min_lat_plot)[0][-1]
max_lat_idx = np.where(lat_vals >= max_lat_plot)[0][-1]
min_lon_idx = np.where(lon_vals >= min_lon_plot)[0][0]
max_lon_idx = np.where(lon_vals >= max_lon_plot)[0][0]
nlat_plot, nlon_plot = np.shape(topo_data[0][max_lat_idx:min_lat_idx,:])
lat_vals_new = np.linspace(min_lat_plot,max_lat_plot,nlat_plot)
lon_vals_new = np.linspace(min_lon_plot,max_lon_plot,nlon_plot)
dx = (lon_vals_new[1]-lon_vals_new[0])/2.
dy = (lat_vals_new[1]-lat_vals_new[0])/2.
extent = [lon_vals_new[0]-dx, lon_vals_new[-1]+dx, lat_vals_new[0]-dy, lat_vals_new[-1]+dy]

fig , ax = plt.subplots()
top = topo_data[0][max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]

from numpy.ma import masked_array
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
top_img = masked_array(top, top == 0.0)
ocn_img = masked_array(top, top != 0.0)
ax.set_facecolor('deepskyblue')
im1 = ax.imshow(top_img,cmap='summer', extent=extent)
import shapely.ops as sops
new_shape = sops.unary_union([el for el in chi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in rus['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in jap['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in nko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in sko['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in phi['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in tai['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
new_shape = sops.unary_union([el for el in vie['geometry'].values[0].geoms])
for geom in new_shape.geoms:
    ax.plot(*geom.exterior.xy, c='k', lw=1.0)
im1 = ax.scatter(training_lons(),training_lats(),color='y',s=10)
im2 = ax.scatter(unseen_lons_new(),unseen_lats_new(),color='r',s=10)
ax.set_xlim([extent[0], extent[1]])
ax.set_ylim([extent[2],extent[3]])
ax.set_ylabel(r'Latitude ($^\circ$N)', labelpad=10)
ax.set_xlabel(r'Longitude ($^\circ$E)', labelpad=10)
plt.legend((im1,im2),('Training epicenters','Unseen epicenters'),loc='upper left')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.tight_layout()
plt.savefig("/Users/emcdugald/sparse_sens_tsunami/geofigs/1028/topo_japan.pdf",dpi=400)
plt.close()




