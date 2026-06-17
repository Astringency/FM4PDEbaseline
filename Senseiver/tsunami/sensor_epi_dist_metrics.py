import os
import scipy.io as sio
import numpy as np
from plot import training_lats, training_lons, unseen_lats_new, unseen_lons_new
import sys
from tabulate import tabulate

logfile = open(os.getcwd()+"/sensor_epi_dist_metrics.out", 'w')
sys.stdout = logfile

swe_data = sio.loadmat(os.getcwd()+"/Data/tsunami/"+"agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_6920_wd_tf_145.mat")
sensor_loc_indices = np.unique(swe_data['sensor_loc_indices'])
sensor_lons = swe_data['longitude'][0][sensor_loc_indices]*180/np.pi
sensor_lats = swe_data['latitude'][0][sensor_loc_indices]*180/np.pi
train_lons = training_lons()
train_lats = training_lats()
unseen_lons = unseen_lons_new()
unseen_lats = unseen_lats_new()


print("5 Closest Sensors for each Training Epicenter")
for i in range(len(train_lons)):
    print("Training Epicenter Coordinate: ({},{})".format(train_lons[i],train_lats[i]))
    lon_diffs = train_lons[i] - sensor_lons
    lat_diffs = train_lats[i] - sensor_lats
    dists = np.sqrt(lon_diffs**2+lat_diffs**2)
    min_dist_idxs = np.argsort(dists)[:5]
    min_dist_lons = sensor_lons[min_dist_idxs]
    min_dist_lats = sensor_lats[min_dist_idxs]
    min_dists = dists[min_dist_idxs]
    table = [["Sensor Coordinate",
              "({},{})".format(min_dist_lons[0],min_dist_lats[0]),
              "({},{})".format(min_dist_lons[1],min_dist_lats[1]),
              "({},{})".format(min_dist_lons[2],min_dist_lats[2]),
              "({},{})".format(min_dist_lons[3],min_dist_lats[3]),
              "({},{})".format(min_dist_lons[4],min_dist_lats[4])],
             ["Distance to Epicenter",
              min_dists[0],
              min_dists[1],
              min_dists[2],
              min_dists[3],
              min_dists[4]]]
    print(tabulate(table))

print("########")
print("########")
print("########")
print("5 Closest Sensors for each Unseen Epicenter")
for i in range(len(unseen_lons)):
    print("Unseen Epicenter Coordinate: ({},{})".format(unseen_lons[i],unseen_lats[i]))
    lon_diffs = unseen_lons[i] - sensor_lons
    lat_diffs = unseen_lats[i] - sensor_lats
    dists = np.sqrt(lon_diffs**2+lat_diffs**2)
    min_dist_idxs = np.argsort(dists)[:5]
    min_dist_lons = sensor_lons[min_dist_idxs]
    min_dist_lats = sensor_lats[min_dist_idxs]
    min_dists = dists[min_dist_idxs]
    table = [["Sensor Coordinate",
              "({},{})".format(min_dist_lons[0],min_dist_lats[0]),
              "({},{})".format(min_dist_lons[1],min_dist_lats[1]),
              "({},{})".format(min_dist_lons[2],min_dist_lats[2]),
              "({},{})".format(min_dist_lons[3],min_dist_lats[3]),
              "({},{})".format(min_dist_lons[4],min_dist_lats[4])],
             ["Distance to Epicenter",
              min_dists[0],
              min_dists[1],
              min_dists[2],
              min_dists[3],
              min_dists[4]]]
    print(tabulate(table))

print("########")
print("########")
print("########")
print("Distances from Unseen Epis to Train Epis")
for i in range(len(unseen_lons)):
    print("Unseen Epicenter Coordinate: ({},{})".format(unseen_lons[i],unseen_lats[i]))
    lon_diffs = unseen_lons[i] - train_lons
    lat_diffs = unseen_lats[i] - train_lats
    dists = np.sqrt(lon_diffs**2+lat_diffs**2)
    table = [["Training Epicenter Coordinate",
              "({},{})".format(train_lons[0],train_lats[0]),
              "({},{})".format(train_lons[1],train_lats[1]),
              "({},{})".format(train_lons[2],train_lats[2]),
              "({},{})".format(train_lons[3],train_lats[3]),
              "({},{})".format(train_lons[4],train_lats[4]),
              "({},{})".format(train_lons[5],train_lats[5]),
              "({},{})".format(train_lons[6],train_lats[6]),
              "({},{})".format(train_lons[7],train_lats[7])],
             ["Distance to Unseen Epicenter",
              dists[0],
              dists[1],
              dists[2],
              dists[3],
              dists[4],
              dists[5],
              dists[6],
              dists[7]]]
    print(tabulate(table))

logfile.close()

