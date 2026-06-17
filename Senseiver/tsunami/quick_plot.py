from glob import glob as gb

import torch

from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver

from plot import plot_all_ts_tsu_japan_unstruct

import multiprocessing
import os

num_gpus = torch.cuda.device_count()
print("Num Devices:", num_gpus)
print("GPU Info:", [torch.cuda.get_device_name(i) for i in range(num_gpus)])

# arg parser
data_config, encoder_config, decoder_config = parse_args()

multiprocessing.set_start_method("fork")

# load the simulation data and create a dataloader
dataloader = senseiver_dataloader(data_config, num_workers=4)

# instantiate new Senseiver
model = Senseiver(
    **encoder_config,
    **decoder_config,
    **data_config
)

# load model (if requested)
if encoder_config['load_model_num'] != None:
    model_num = encoder_config['load_model_num']
    print(f'Loading {model_num} ...')

    model_loc = gb(f"lightning_logs/version_{model_num}/checkpoints/*.ckpt")[0]
    # Use the below commented code if using on HPC
    # model = Senseiver.load_from_checkpoint(model_loc,
    #                                        **encoder_config,
    #                                        **decoder_config,
    #                                        **data_config)
    model = Senseiver.load_from_checkpoint(model_loc, map_location=torch.device('cpu'),
                                           **encoder_config,
                                           **decoder_config,
                                           **data_config)

path = model_loc.split('checkpoints')[0]
name = 'tensor.pt'
unseen_flag = data_config['unseen_flag']
mat_data = data_config['data_key']

with torch.no_grad():
    if not (data_config['path_pref']=='for_paper' in data_config['path_pref']):
        if data_config['unseen_flag'] == True:
            output_im = torch.load(os.getcwd()+"/lightning_logs/version_"+str(model_num)+'/tensor'+'_unseen'+'.pt')
        else:
            output_im = torch.load(os.getcwd() + "/lightning_logs/version_" + str(model_num) + '/tensor' + '_training' + '.pt')

    else:
        if data_config['unseen_flag'] == True:
            output_im = torch.load(os.getcwd()+"/"+data_config['path_pref']+"/version_"+str(model_num)+'/tensor'+'_unseen'+'.pt')
        else:
            output_im = torch.load(os.getcwd()+"/"+data_config['path_pref']+"/version_" + str(model_num) + '/tensor' + '_training' + '.pt')

if data_config['path_pref']=='for_paper' in data_config['path_pref']:
    path = path.replace('lightning_logs', data_config['path_pref'])

if unseen_flag:
    name = 'tensor_unseen.pt'
    type = 'unseen'
else:
    name = 'tensor_training.pt'
    type = 'training'
plot_all_ts_tsu_japan_unstruct(dataloader, output_im, mat_data, type, path)


