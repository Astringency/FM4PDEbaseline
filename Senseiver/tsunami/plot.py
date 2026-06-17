#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"]=16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

def training_lons():
    return np.array([136.6180, 139.5560, 139.3290, 138.9350, 140.9290, 135.7400, 141.5010, 142.3870])

def training_lats():
    return np.array([33.0700, 28.8560, 28.9320, 29.3840, 33.4530, 33.1570, 35.9360, 35.2670])

def unseen_lons():
    return np.array([136.6500, 140.2000, 138.9000, 139.5000])

def unseen_lats():
    return np.array([33.1000, 29.1000, 28.1000, 28.8000])


def unseen_lons_new():
    return np.array([136.6500,138.2000,138.9000,
                     139.5000,140.2000,140.5000,
                     141.5000,142.5000])

def unseen_lats_new():
    return np.array([33.1000,31.0000,28.1000,
                     28.8000,29.1000,31.8000,
                     34.2000,36.2000])



def plot_err_for_each_epi(means, matname, times, dta_type, path, plot_pts):
    time_idxs = np.insert(times,0,0)
    pts = np.sort(plot_pts)

    if dta_type=='training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()

    j=0
    for i in range(1, len(time_idxs)):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        emean = means[int(time_idxs[i - 1]):int(time_idxs[i])]
        epi_lon = lons[j]
        epi_lat = lats[j]
        if dta_type == 'training':
            start_plt_pts = (i - 1) * (np.shape(emean)[0])
            end_plt_pts = (i) * (np.shape(emean)[0])
            plot_points = pts[(pts > start_plt_pts) & (pts < end_plt_pts)] - start_plt_pts
            ax.plot(emean)
            ax.scatter(plot_points*(5./6.), torch.tensor(emean)[plot_points], c='r', marker='o', s=5)
            ax.title.set_text(f'Average Error = {torch.tensor(emean).mean():0.3}')
            ax.set(xlabel="Time Steps", ylabel="$|true-pred|/\max(|true|)$")
            ax.legend(['all data', 'training data', '.10 threshold', '.05 threshold'])
        else:
            plot_pts = np.arange(0,len(emean),1)*(5./6.)
            plot_pts = plot_pts
            ax.plot(plot_pts,emean)
            ax.title.set_text(f'Average Error = {torch.tensor(emean).mean():0.3}')
            ax.scatter(plot_pts[15],emean[15],s=75,marker="x",color='r')
            ax.scatter(plot_pts[85], emean[85], s=75, marker="x", color='g')
            ax.set(xlabel="Time (minutes)", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            ax.legend(['Error from $0$-$2$ hours', r'Error at $12.5$ mins: ${}$'.format(round(emean[16],3)), r'Error at $70.83$ mins: ${}$'.format(round(emean[123],3))])

        j += 1


        plt.tight_layout()
        if path:
            plt.savefig(path + '/1028_all_ts_{}'.format(dta_type)+"_epi_{}_{}".format(epi_lon,epi_lat)+"_v3.pdf",dpi=400,bbox_inches='tight')
            plt.close()



def plot_timeseries_for_each_epi_unstruct(true,pred,times,sens_idxs,sens_locs,dta_type,path):
    time_idxs = np.insert(times, 0, 0)

    if dta_type == 'training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()


    for i in range(1, len(time_idxs)):
        j = 0
        for idx in sens_idxs[0]:
            true_ts = true[int(time_idxs[i - 1]):int(time_idxs[i]), idx]
            pred_ts = pred[int(time_idxs[i - 1]):int(time_idxs[i]), idx]
            if true_ts.abs().max() >= 1e-3:
                times = np.arange(0,len(true_ts),1)*(5/6)
                times = times + 120.0
                fig, ax = plt.subplots()
                ax.plot(times,true_ts, c='g')
                ax.plot(times,pred_ts, c='k')

                lon_deg = sens_locs[j][0] * 180 / np.pi
                lat_deg = sens_locs[j][1] * 180 / np.pi

                ax.title.set_text('Wave Height Time Series at sensor ({}$^\circ$E, {}$^\circ$N)'.format(round(lon_deg,1), round(lat_deg,1)))
                ax.set(xlabel="Time (minutes)", ylabel="Wave Height (m)")
                ax.legend(['True Height', 'Predicted Height'])

                plt.tight_layout()

                plt.savefig(path + 'timeseries/{}'.format(dta_type)
                + "/sens_{}_{}_epi_{}_{}".format(round(lon_deg,3),round(lat_deg,3),round(lons[i-1],3),round(lats[i-1],3)) + "_1028.pdf",dpi=400,bbox_inches='tight')
                plt.close()

            j += 1




def plot_div_err_for_each_epi(means, swemeans, times, dta_type, path):
    for k in range(len(times)):
        times[k] -= (k+1)
    time_idxs = np.insert(times,0,0)

    if dta_type=='training':
        lons = training_lons()
        lats = training_lats()
    else:
        lons = unseen_lons_new()
        lats = unseen_lats_new()

    j=0
    for i in range(1, len(time_idxs)):
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6,3))
        emean = means[int(time_idxs[i - 1])+8:int(time_idxs[i])]
        swe_emean = swemeans[int(time_idxs[i - 1])+8:int(time_idxs[i])]
        times = np.arange(0,len(emean),1)*(5/6)
        times = times + 120.
        epi_lon = lons[j]
        epi_lat = lats[j]
        ax.plot(times, emean, c='g')
        ax.plot(times, swe_emean, c='k')
        ax.title.set_text(f'Average $h_t$ error = {torch.tensor(emean).mean():0.3}')
        ax.set(xlabel="Time (minutes)", ylabel=r"$\frac{|h_t-\hat{h_t}|}{\max(|h|)}$")
        ax.legend(['Tsunseiver Error', 'SWE Error'])

        j += 1
        plt.tight_layout()
        if path:
            plt.savefig(path + '/1028_all_ts_div_err_{}'.format(dta_type)+"_epi_{}_{}".format(epi_lon,epi_lat)+".pdf",dpi=400,bbox_inches='tight')
            plt.close()

def plot_all_ts_tsu_japan_unstruct(dataloader, pred, mat, data_type,  path=None):
    with torch.no_grad():

        true = dataloader.dataset.data.to('cpu')
        pred = pred.to('cpu')

        if len(true.shape) == 5:
            dims = (1, 2, 3)
        if len(true.shape) == 4:
            dims = (1, 2)
        if len(true.shape) == 3:
            dims = (1)

        mask = dataloader.dataset.mask
        times = true.size()[0]

        if len(true.shape) == 4:
            full_mask = mask.repeat([times,1,1])
        else:
            full_mask = mask.repeat([times,1])

        true_masked = torch.where((full_mask == 0), true[..., 0], 0)
        pred_masked = torch.where((full_mask == 0), pred[...,0], 0)
        maxes = [true_masked[i, :].abs().max() for i in range(len(true_masked))]

        abs_errs = [(true_masked[i, :] - pred_masked[i, :]).abs() for i in
                    range(len(true_masked))]
        emax = [(abs_errs[i] / maxes[i]).max().item() for i in range(len(abs_errs))]
        ratios = [(abs_errs[i] / maxes[i]) for i in range(len(abs_errs))]
        emean = [ratios[i].flatten()[torch.where(true_masked[i].flatten().abs() > 1e-4)].mean().item() for i in range(len(abs_errs))]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_points = dataloader.dataset.train_ind
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        if data_type == 'training':
            ax.plot(emean)
            ax.scatter(plot_points, torch.tensor(emean)[plot_points.long()], c='r', marker='o', s=5)
            ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).nanmean():0.3}')
            ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            ax.vlines(dataloader.dataset.time_idx.astype(int), 0, np.max(emean), color='g')
            ax.legend(['all data', 'training data','epi split','.10 threshold','.05 threshold'])

        else:
            ax.plot(emean)
            ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).nanmean():0.3}')
            ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            ax.vlines(dataloader.dataset.time_idx.astype(int), 0, np.max(emean), color='g')
            ax.legend(['all data', 'epi split','.10 threshold', '.05 threshold'])

        plt.tight_layout()
        if path:
            plt.savefig(path + '/1028_all_ts_{}'.format(data_type)+".png")
            plt.close()

        print(f'The mean mean err is {torch.tensor(emean).nanmean():0.3}')
        print(f'The mean max err is {torch.tensor(emax).nanmean():0.3}')

        sim_times = dataloader.dataset.time_idx
        plot_err_for_each_epi(emean, mat, sim_times, data_type, path, plot_points)

        mat_data = sio.loadmat(os.getcwd()+"/Data/tsunami/"+mat)
        sensor_loc_indices = mat_data['sensor_loc_indices']
        sensor_locs = mat_data['sensor_locs']

        if not os.path.exists(path + 'timeseries/{}'.format(data_type)):
            if not os.path.exists(path + 'timeseries/'):
                os.mkdir(path + 'timeseries/')
            os.mkdir(path + 'timeseries/{}'.format(data_type))

        j = 0
        for idx in sensor_loc_indices[0]:
            true_ts = true[:,idx]
            pred_ts = pred[:,idx]

            if true_ts.abs().max() >= 1e-1:
                fig, ax = plt.subplots()
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                ax.plot(true_ts,c='g')
                ax.plot(pred_ts,c='k')
                ax.plot((true_ts - pred_ts).abs(), c='r')

                lon_deg = sensor_locs[j][1] * 180 / np.pi - 180
                lat_deg = sensor_locs[j][0] * 180 / np.pi
                ax.title.set_text('Wave Height Time Series at {}, {}'.format(round(lon_deg),round(lat_deg)))
                ax.set(xlabel="Frame Index", ylabel="Normalized Wave Height (m)")
                ax.legend(['True Height', 'Predicted Height', 'Absolute Error'])

                plt.tight_layout()

                plt.savefig(path + 'timeseries/{}'.format(data_type)+"/1028_sens_{}_{}_all_epis".format(round(lon_deg),round(lat_deg))+".png")
                plt.close()

            j += 1

        plot_timeseries_for_each_epi_unstruct(true, pred, sim_times, sensor_loc_indices, sensor_locs, data_type, path)

        if 'div' in dataloader.dataset.data_name:
            n_sims = len(dataloader.dataset.time_idx)
            train_ind_start = np.append(0, dataloader.dataset.time_idx[0:n_sims - 1])
            train_inds = torch.concatenate([torch.range(train_ind_start[i], dataloader.dataset.time_idx[i] - 2) for i in range(len(dataloader.dataset.time_idx))])
            curr_inds = torch.add(train_inds, 1).long()
            prev_inds = train_inds.long()
            pred_dhdt = (pred[curr_inds,:,:]-pred[prev_inds,:,:])/50
            swe_dhdt = (true[curr_inds,:,:]-true[prev_inds,:,:])/50
            true_div = dataloader.dataset.divu.to('cpu')
            true_div_prev = true_div[prev_inds,:,:]
            true_div_curr = true_div[curr_inds,:,:]
            true_div = -.5*(true_div_prev+true_div_curr)
            times = true_div.size()[0]
            if len(true.shape) == 4:
                full_mask = mask.repeat([times, 1, 1])
            else:
                full_mask = mask.repeat([times, 1])
            true_div_masked = torch.where((full_mask == 0), true_div[..., 0], 0)
            pred_dhdt_masked = torch.where((full_mask == 0), pred_dhdt[..., 0], 0)

            swe_dhdt_masked = torch.where((full_mask == 0), swe_dhdt[..., 0], 0)

            maxes = [true_div_masked[i, :].abs().max() for i in range(len(true_div_masked))]
            abs_errs = [(true_div_masked[i, :] - pred_dhdt_masked[i, :]).abs() for i in
                        range(len(true_div_masked))]
            ratios = [(abs_errs[i] / maxes[i]) for i in range(len(abs_errs))]
            emean = [ratios[i].flatten()[torch.where(ratios[i].flatten().abs() > 1e-8)].mean() for i in
                     range(len(abs_errs))]

            swe_abs_errs = [(true_div_masked[i, :] - swe_dhdt_masked[i, :]).abs() for i in
                        range(len(true_div_masked))]
            swe_emax = [(swe_abs_errs[i] / maxes[i]).max().item() for i in range(len(swe_abs_errs))]
            swe_ratios = [(swe_abs_errs[i] / maxes[i]) for i in range(len(swe_abs_errs))]
            swe_emean = [swe_ratios[i].flatten()[torch.where(true_masked[i].flatten().abs() > 1e-4)].mean() for i in
                     range(len(abs_errs))]

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.plot(emean)
            ax.scatter(range(len(swe_emean)),swe_emean, c='r',s=.5)
            ax.title.set_text(f'$\epsilon$ = {torch.tensor(emean).mean():0.3}')
            ax.set(xlabel="Frame Index", ylabel=r"$\frac{|h-\hat{h}|}{\max(|h|)}$")
            ax.vlines(dataloader.dataset.time_idx.astype(int), 0, max(np.max(swe_emean),np.max(emean)), color='g')
            ax.legend(['Tsunseiver Err', 'SWE Err', 'epi split'])

            plt.tight_layout()
            if path:
                plt.savefig(path + '/1028_divergence_all_ts_{}'.format(data_type) + ".png")
                plt.close()

            print(f'The mean mean div err is {torch.tensor(emean).mean():0.3}')
            print(f'The mean max div err is {torch.tensor(swe_emax).mean():0.3}')

            sim_times = dataloader.dataset.time_idx
            plot_div_err_for_each_epi(emean, swe_emean, sim_times, data_type, path)





