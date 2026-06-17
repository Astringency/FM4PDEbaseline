import scipy.io as sio
import os


def tsunami_sensors(fname):
    fpath = os.getcwd() + "/Data/tsunami/" + fname
    data = sio.loadmat(fpath)
    sensor_loc_inds = data['sensor_loc_indices']
    return sensor_loc_inds

def structTsunami_sensors(fname):
    fpath = os.getcwd() + "/Data/tsunami/" + fname
    data = sio.loadmat(fpath)
    sensor_loc_inds = data['sensor_loc_indices']
    return sensor_loc_inds[:,0], sensor_loc_inds[:,1]
                
        
        
    
    

        
        
    
    
    
