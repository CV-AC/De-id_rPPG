import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from torchvision import models
import imageio.v2 as imageio
import pickle
from load_cropped_blocks_rppg_pure import load_cropped_pure
from torch.utils.data import DataLoader
from utils.Average_Accuracy import AverageMeter
from utils.Average_Accuracy import Acc
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as op
import time
import random
from scipy.fft import rfft
from scipy import signal
import math
import os
import argparse
import glob
from Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
import torch.backends.cudnn as cudnn
from scipy.signal import butter, filtfilt, resample, sosfiltfilt, welch, detrend
from scipy.fft import fft
from utils.signals_stuff import butter_bandpass, norm, hr_fft,NegPearson
import sys

loss_global_train = []
loss_global_valid = []
acc_global_train = []
acc_global_valid = []

def train(train_loader, model, criterion, optimizer, epoch, name_of_run,whatdb):

    #Run one train epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = Acc()

    # switch to train mode
    model.train()
    list_gt = []
    list_pred = []
    dict_gt = {}
    dict_pred = {}

    list_names = []
    dict_signals = {}
    dict_bvps = {}

    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        # compute output
        output = model(input,2.0)
        output = (output - torch.mean(output)) / torch.std(output)
        bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize
        loss = criterion(output,bvp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        list_pred_f = []
        list_pred_pxx = []
        list_gt_f = []
        list_gt_pxx = []
        for b in range(0,output.size()[0]):
            if whatdb == "mmse":
                pred, pred_pxx, pred_f, _ = hr_fft(output[b].detach().cpu().numpy(),fs=25)
                gt, gt_pxx, gt_f, _ = hr_fft(bvp[b].detach().cpu().numpy(),fs=25)
            else:
                pred, pred_pxx, pred_f, _ = hr_fft(output[b].detach().cpu().numpy())
                gt, gt_pxx, gt_f, _ = hr_fft(bvp[b].detach().cpu().numpy())
            list_pred_f.append(pred_f)
            list_pred_pxx.append(pred_pxx)
            list_gt_f.append(gt_f)
            list_gt_pxx.append(gt_pxx)
            error = pred-gt
            list_gt.append(gt)
            list_pred.append(pred)
            acc.update(error, 1)

            temp = output[b].detach().cpu().numpy()
            tempbvp = bvp[b].detach().cpu().numpy()
            dirtemp = dir[b]
            dirtemp = dirtemp.split("/")[-2]

            frame0temp = str(frame0[b].detach().cpu().numpy())
            frame0temp = frame0temp.zfill(4)
            name = dirtemp+"__"+frame0temp

            list_names.append(name)
            dict_gt[name] = gt
            dict_pred[name] = pred
            dict_signals[name] = temp
            dict_bvps[name] = tempbvp

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(train_loader)-2:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {acc.mae:.4f}\t'
                    'RMSE {acc.rmse:.4f}\t'
                    'STD {acc.std:.4f}\n'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,acc=acc))
            BS = output.size()[0]
            if BS > 1:
                fig,ax = plt.subplots(BS,2,figsize=(15,3))
            else:
                fig,ax = plt.subplots(2,2,figsize=(15,3))
            for index in range(0,BS):
                ax[index][0].plot(norm(output[0].detach().cpu().numpy()))
                ax[index][0].plot(norm(bvp[0].detach().cpu().numpy()))
                ax[index][1].plot(list_pred_f[index],norm(list_pred_pxx[index]))
                ax[index][1].plot(list_gt_f[index],norm(list_gt_pxx[index]))
                ax[index][1].set_xlim(0,3)
                ax[index][0].set_title("Pred: "+str(list_pred[-BS+index]))
                ax[index][1].set_title("GT: "+str(list_gt[-BS+index]))
            plt.savefig("./Images_"+name_of_run+"/"+"Train_"+str(epoch)+"_"+str(i)+".svg")
            plt.close()

            with open(name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {acc.mae:.4f}\t'
                        'RMSE {acc.rmse:.4f}\t'
                        'STD {acc.std:.4f}\n'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses,acc=acc))

    fig,ax = plt.subplots()
    loss_global_train.append(losses.avg)
    acc_global_train.append(acc.mae)

    list_gt = np.array(list_gt)
    list_pred = np.array(list_pred)
    errors = np.abs(list_gt-list_pred)

    fig,ax = plt.subplots(3,1)
    ax[0].set_xlim([0,300])
    ax[1].set_xlim([0,300])
    ax[2].set_xlim([0,300])
    ax[0].hist(list_pred,50)
    ax[1].hist(list_gt,50)
    ax[2].plot(errors)
    if not os.path.exists("./Images_"+name_of_run):
        os.makedirs("./Images_"+name_of_run)
    plt.savefig("./Images_"+name_of_run+"/"+"Train: Epoch_"+str(epoch)+".png")
    plt.close()

def validate(valid_loader, model, criterion, epoch, name_of_run,whatdb):

    #Run one eval epoch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = Acc()
    list_gt = []
    list_pred = []
    dict_gt = {}
    dict_pred = {}

    list_names = []
    dict_signals = {}
    dict_bvps = {}
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input,bvp,id,gender,race,dir,frame0,lnd,mask)  in enumerate(valid_loader):
        data_time.update(time.time() - end)
        input = input.to(device=device, dtype=torch.float)
        bvp = bvp.to(device=device, dtype=torch.float)
        with torch.no_grad():
            # compute output
            output = model(input,2.0)
            output = (output - torch.mean(output)) / torch.std(output)
            bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)
            loss = criterion(output,bvp)
            loss = loss.float()

        list_pred_f = []
        list_pred_pxx = []
        list_gt_f = []
        list_gt_pxx = []
        for b in range(0,output.size()[0]):
            if whatdb == "mmse":
                pred, pred_pxx, pred_f, pred_filtsig = hr_fft(output[b].detach().cpu().numpy(),fs=25)
                gt, gt_pxx, gt_f, gt_filtsig = hr_fft(bvp[b].detach().cpu().numpy(),fs=25)
            else:
                pred, pred_pxx, pred_f, pred_filtsig = hr_fft(output[b].detach().cpu().numpy())
                gt, gt_pxx, gt_f, gt_filtsig = hr_fft(bvp[b].detach().cpu().numpy())
            list_pred_f.append(pred_f)
            list_pred_pxx.append(pred_pxx)
            list_gt_f.append(gt_f)
            list_gt_pxx.append(gt_pxx)
            error = abs(pred-gt)
            list_gt.append(gt)
            list_pred.append(pred)
            acc.update(error, 1)

            temp = output[b].detach().cpu().numpy()
            tempbvp = bvp[b].detach().cpu().numpy()
            dirtemp = dir[b]
            dirtemp = dirtemp.split("/")[-2]

            frame0temp = str(frame0[b].detach().cpu().numpy())
            frame0temp = frame0temp.zfill(4)
            name = dirtemp+"__"+frame0temp

            list_names.append(name)
            dict_gt[name] = gt
            dict_pred[name] = pred
            dict_signals[name] = temp
            dict_bvps[name] = tempbvp
            #np.save("./output_signals/"+dirtemp+"__"+frame0temp,temp )

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 or i == len(valid_loader)-2:
            print('Valid: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {acc.mae:.4f}\t'
                    'RMSE {acc.rmse:.4f}\t'
                    'STD {acc.std:.4f}\n'.format(
                      epoch, i, len(valid_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,acc=acc))

            with open(name_of_run+".txt", "a") as file_object:
            # Append 'hello' at the end of file
                file_object.write('Valid: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {acc.mae:.4f}\t'
                        'RMSE {acc.rmse:.4f}\t'
                        'STD {acc.std:.4f}\n'.format(
                          epoch, i, len(valid_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses,acc=acc))

            BS = output.size()[0]
            if BS > 1:
                fig,ax = plt.subplots(BS,2,figsize=(15,3))
            else:
                fig,ax = plt.subplots(2,2,figsize=(15,3))
            for index in range(0,BS):
                ax[index][0].plot(norm(pred_filtsig))
                ax[index][0].plot(norm(gt_filtsig))
                ax[index][1].plot(list_pred_f[index],norm(list_pred_pxx[index]))
                ax[index][1].plot(list_gt_f[index],norm(list_gt_pxx[index]))
                ax[index][1].set_xlim(0,3)
                ax[index][0].set_title("Pred: "+str(list_pred[-BS+index]))
                ax[index][1].set_title("GT: "+str(list_gt[-BS+index]))
            plt.savefig("./Images_"+name_of_run+"/"+"Valid_"+str(epoch)+"_"+str(i)+".svg")
            plt.close()

    loss_global_valid.append(losses.avg)
    acc_global_valid.append(acc.mae)

    list_gt = np.array(list_gt)
    list_pred = np.array(list_pred)
    errors = np.abs(list_gt-list_pred)

    fig,ax = plt.subplots(3,1)
    ax[0].set_xlim([0,300])
    ax[1].set_xlim([0,300])
    ax[2].set_xlim([0,300])
    ax[0].hist(list_pred,50)
    ax[1].hist(list_gt,50)
    ax[2].plot(errors)
    if not os.path.exists("./Images_"+name_of_run):
        os.makedirs("./Images_"+name_of_run)
    plt.savefig("./Images_"+name_of_run+"/"+"Valid: Epoch_"+str(epoch)+".png")
    plt.close()

whatdb = str(sys.argv[1])


BATCH_SIZE = 8
NUM_WORKERS = 0
lr = 1e-4
momentum = 0.9
wd = 5e-4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

seq_len = 64
model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(seq_len,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
model = nn.DataParallel(model)
model = model.to(device)
cudnn.benchmark = True

transformz = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

pklfile = open("./folds/"+whatdb+"_folds_rppg.pkl",'rb')

train_samples = pickle.load(pklfile)
valid_samples = pickle.load(pklfile)

if whatdb == "pure":
    train_dataset = load_cropped_pure(data=train_samples,shuffle=True, Training = True, transform=transformz)
    valid_dataset = load_cropped_pure(data=valid_samples,shuffle=False, Training = False, transform=transformz)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)
valid_loader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

criterion = NegPearson()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
#optimizer = op.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),lr=lr, weight_decay=0.05) #from SWIN paper change to resnet optimizer
name_of_run = "Physformer_"+whatdb

if not os.path.exists("./Images_"+name_of_run):
    os.makedirs("./Images_"+name_of_run)

total_epochs = 10
if whatdb == "pure" or whatdb == "mmse":
    total_epochs = 20
for epoch in range(0, total_epochs):
    train(train_loader, model, criterion,optimizer,epoch,name_of_run,whatdb)
    validate(valid_loader,model, criterion,epoch,name_of_run,whatdb)

    if len(loss_global_valid)>1:
        if loss_global_valid[-1] <= min(loss_global_valid[:-1]):
            print("Save")
            torch.save(model.state_dict(), name_of_run+".pt")

    fig,ax = plt.subplots(1,1)
    plt.plot(loss_global_train,'y-')
    plt.plot(loss_global_valid, 'b-')
    plt.grid()
    plt.savefig(name_of_run+"_loss")
    plt.close()

    fig,ax = plt.subplots(1,1)
    plt.plot(acc_global_train,'y-')
    plt.plot(acc_global_valid, 'b-')
    plt.grid()
    plt.savefig(name_of_run+"_acc")
    plt.close()
