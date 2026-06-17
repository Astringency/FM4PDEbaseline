#!/bin/bash -l
#SBATCH --output=no_div_tsu_2hr.out
#SBATCH --partition=shared-gpu
#SBATCH --time=0-10:00:00
#SBATCH --nodelist=cn4042

#For the training data, use data-key agg_8_sims_0_time_ss_4_ss_unstruct_ntimes_1564_wd.mat
#For the unseen data, use data-key agg_4_sims_0_time_ss_4_ss_unstruct_ntimes_802_wd.mat
#Set the unseen_flag for plotting purposes
module load conda
conda activate torch_gpu_0214
srun python train.py --data_name tsunami_with_div --data_key unseen_agg_8_sims_0_time_ss_2_ss_unstruct_ntimes_3464_wd_new_epis_tf_145.mat --training_frames 928 --cons False --seed 123 --enc_preproc 128 --dec_num_latent_channels 64 --enc_num_latent_channels 64 --num_latents 512 --dec_preproc_ch 64 --test True --local_flag False --num_layers 3 --batch_pixels 4096 --batch_frames 100 --num_dims 3 --space_bands 32 --unseen_flag True --bath_sampling 100 --lam 0.00 --load_model_num 0
