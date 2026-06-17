import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import torch
import math

from datasets import SWETsunamiWdiv

from sensor_loc import tsunami_sensors



import datetime
from positional import UnstructuredPositionalEncoderWbath



from torch.utils.data import DataLoader,Dataset






def load_data(dataset_name, num_sensors, seed=123,data_key=None):
    if 'agg' in data_key:
        data, lats, longs, ocn_floor, divu, mask, data_idx = SWETsunamiWdiv(data_key)
        x_sens, *y_sens = tsunami_sensors(data_key)
        return torch.as_tensor(data, dtype=torch.float), x_sens, y_sens, lats, longs, ocn_floor, torch.as_tensor(mask, dtype=torch.int), torch.as_tensor(divu, dtype=torch.float), data_idx
    else:
        data, lats, longs, ocn_floor, divu, mask = SWETsunamiWdiv(data_key)
        x_sens, *y_sens = tsunami_sensors(data_key)
        return torch.as_tensor(data, dtype=torch.float), x_sens, y_sens, lats, longs, ocn_floor, torch.as_tensor(mask, dtype=torch.int), torch.as_tensor(divu, dtype=torch.float)
    
    
    
def senseiver_dataloader(data_config, num_workers=0):
    return DataLoader( senseiver_loader(data_config), batch_size=None, 
                       pin_memory=True, 
                       shuffle = True,
                       # hacky way of avoid workers>0 crashing on Mac
                        num_workers=4
                       #num_workers=[0 if os.environ['_'].find('MacOS')>-1 else 4][0]
                     )
    

class senseiver_loader(Dataset):
    
    def __init__(self,  data_config):
    
        data_name   = data_config['data_name']
        num_sensors = data_config['num_sensors']
        seed        = data_config['seed']
        self.data_name = data_name
        data_key = data_config['data_key']
        if 'agg' in data_key:
            self.data, x_sens, y_sens, lats, longs, ocn_floor, self.mask, self.divu, self.time_idx = load_data(
                data_name, num_sensors, seed, data_key=data_key)
        else:
            self.data, x_sens, y_sens, lats, longs, ocn_floor, self.mask, self.divu = load_data(
                data_name, num_sensors, seed, data_key=data_key)
        total_frames, *image_size, im_ch = self.data.shape
        data_config['total_frames'] = total_frames
        data_config['image_size']   = image_size
        data_config['im_ch']        = im_ch
        self.training_frames = data_config['training_frames']
        self.batch_frames    = data_config['batch_frames'] 
        self.batch_pixels    = data_config['batch_pixels']
        num_batches = int(self.data.shape[1:].numel()*self.training_frames/(
                                            self.batch_frames*self.batch_pixels))
        
        assert num_batches>0
        
        print(f'{num_batches} Batches of data per epoch\n')
        data_config['num_batches'] = num_batches
        self.num_batches = num_batches
        
        if data_config['consecutive_train']:
            self.train_ind = torch.arange(0,self.training_frames)
        else:
            if seed:
                torch.manual_seed(seed)
            if 'div' in data_name:
                if 'agg' in data_key:
                    n_sims = len(self.time_idx)
                    train_ind_start = np.append(0,self.time_idx[0:n_sims-1])
                    train_inds = torch.concatenate([torch.range(train_ind_start[i],self.time_idx[i]-2) for i in range(len(self.time_idx))])
                    self.train_ind = train_inds[torch.randperm(len(train_inds))]
                    self.train_ind = self.train_ind[:self.training_frames]
                else:
                    self.train_ind = torch.randperm(self.data.shape[0]-1)[:self.training_frames]
            else:
                self.train_ind = torch.randperm(self.data.shape[0])[:self.training_frames]
            print(self.train_ind)
            
        if self.batch_frames > self.training_frames:
            print('Warning: batch_frames bigger than num training samples')
            self.batch_frames = self.training_frames
            
        # sensor coordinates
        sensors = np.zeros(self.data.shape[1:-1])
        
        if len(sensors.shape) == 2:
            sensors[x_sens,y_sens] = 1
        elif len(sensors.shape) == 3: # 3D images
            sensors[x_sens,y_sens[0],y_sens[1]] = 1
        else:
            sensors[x_sens] = 1
            
        self.sensors,*_ = np.where(sensors.flatten()==1)
        
        # sine-cosine positional encodings
        data_len = self.data.shape[1]
        lat_max_freq = math.ceil(np.sqrt(data_len/2))
        lon_max_freq = int(2 * lat_max_freq)
        ocn_floor_max_freq = data_config['bath_sampling']
        self.pos_encodings = UnstructuredPositionalEncoderWbath(self.data.shape[1:],
                                                           data_config['space_bands'],
                                                           lats, longs, ocn_floor,
                                                           max_frequencies=[lat_max_freq, lon_max_freq, ocn_floor_max_freq])
        self.indexed_sensors  = self.data.flatten(start_dim=1, end_dim=-2)[:,self.sensors,]
        self.sensor_positions = self.pos_encodings[self.sensors,]
        self.sensor_positions = self.sensor_positions[None,].repeat_interleave(
                                                    self.batch_frames, axis=0)
        self.pix_avail = torch.where(self.mask.flatten()==0)[0]
        if seed:
            torch.manual_seed(datetime.datetime.now().microsecond) # reset seed
            
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        frames = self.train_ind[torch.randperm(self.training_frames)][:self.batch_frames]
        pixels = self.pix_avail[torch.randperm(*self.pix_avail.shape)][:self.batch_pixels]
        sensor_vals = self.indexed_sensors[torch.add(frames, 1).long(),]
        sensor_vals = torch.cat([sensor_vals, self.sensor_positions], axis=-1)
        coords = self.pos_encodings[pixels,][None,]
        coords = coords.repeat_interleave(self.batch_frames, axis=0)
        field_prev = self.data.flatten(start_dim=1, end_dim=-2)[frames.long(),][:,pixels,]
        field_curr = self.data.flatten(start_dim=1, end_dim=-2)[torch.add(frames,1).long(),][:,pixels,]
        div_prev = self.divu.flatten(start_dim=1, end_dim=-2)[frames.long(),][:,pixels,]
        div_curr = self.divu.flatten(start_dim=1, end_dim=-2)[torch.add(frames,1).long(),][:,pixels,]
        return sensor_vals, coords, field_prev, field_curr, div_prev, div_curr
        
     
    
