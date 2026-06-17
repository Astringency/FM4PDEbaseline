from glob import glob as gb

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from s_parser import parse_args
from dataloaders import senseiver_dataloader
from network_light import Senseiver

from plot import plot_all_ts_tsu_japan_unstruct

import multiprocessing

num_gpus = torch.cuda.device_count()
print("Num Devices:",num_gpus)
print("GPU Info:",[torch.cuda.get_device_name(i) for i in range(num_gpus)])




# arg parser
data_config, encoder_config, decoder_config = parse_args()

if data_config['local_flag']:
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
    #Use the below commented code if using on HPC
    # model = Senseiver.load_from_checkpoint(model_loc,
    #                                        **encoder_config,
    #                                        **decoder_config,
    #                                        **data_config)
    model = Senseiver.load_from_checkpoint(model_loc, map_location=torch.device('cpu'),
                                       **encoder_config,
                                       **decoder_config,
                                       **data_config)
else:
    model_loc = None

if not data_config['test']:
    # callbacks
    if data_config['local_flag']:
        cbs = [ModelCheckpoint(monitor="train_loss", filename="train-{epoch:02d}",
                               every_n_epochs=1, save_on_train_epoch_end=True),
               EarlyStopping(monitor="train_loss", check_finite=False, stopping_threshold=1e-8)]

        trainer = Trainer(max_epochs=-1,
                          callbacks=cbs,
                          #gpus=data_config['gpu_device'],
                          #accumulate_grad_batches=data_config['accum_grads'],
                          log_every_n_steps=data_config['num_batches'],
                          )
    else:
        cbs = [ModelCheckpoint(monitor="train_loss", filename="train-{epoch:02d}",
                               every_n_epochs=10, save_on_train_epoch_end=True),
               EarlyStopping(monitor="train_loss", check_finite=False, patience=1000)]

        trainer = Trainer(precision=16,
                          #precision=8,
                          max_epochs=-1,
                          callbacks=cbs,
                          accelerator='auto',
                          accumulate_grad_batches=1,
                          log_every_n_steps=data_config['num_batches'],
                          )

    # device = torch.device("cuda:0")
    # if torch.cuda.device_count() > 1:
    #     print("Using Multi GPU")
    #     model = nn.DataParallel(model)
    # else:
    #     print("Only 1 GPU available")
    #     model = model.to(device)

    trainer.fit(model, dataloader, ckpt_path=model_loc)

else:
    if data_config['gpu_device']:
        device = data_config['gpu_device'][0]
        model = model.to(f"cuda:{device}")
        
        model = model.to(f"cuda:{data_config['gpu_device'][0]}")
        dataloader.dataset.data = torch.as_tensor(dataloader.dataset.data).to(f"cuda:{device}")
        dataloader.dataset.sensors = torch.as_tensor(dataloader.dataset.sensors).to(f"cuda:{device}")
        dataloader.dataset.pos_encodings = torch.as_tensor(dataloader.dataset.pos_encodings).to(f"cuda:{device}")
        
    path = model_loc.split('checkpoints')[0]
    name = 'tensor.pt'
    unseen_flag = data_config['unseen_flag']
    mat_data = data_config['data_key']

    if data_config['path_pref']=='for_paper' in data_config['path_pref']:
        path = path.replace('lightning_logs', data_config['path_pref'])


    with torch.no_grad():
        output_im = model.test(dataloader, num_pix=2048, split_time=100)
    if unseen_flag:
        name = 'tensor_unseen.pt'
        type = 'unseen'
    else:
        name = 'tensor_training.pt'
        type = 'training'
    plot_all_ts_tsu_japan_unstruct(dataloader, output_im, mat_data, type, path)
    torch.save(output_im, f'{path}/' + name)

