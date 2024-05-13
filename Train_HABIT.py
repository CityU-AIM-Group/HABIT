from __future__ import print_function

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import random
import time
import argparse
import os
import sys
import ast
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import BatchNorm2d
from sklearn.cluster import KMeans

from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader

from utils import accuracy, compute_mAP, setup_default_logging, AverageMeter, WarmupCosineLrScheduler, visualizaion

import torchvision.models as models
from models.densenet import densenet121, densenet161
from models import ResMLP, MLPMixer
from models.mlp_mixer import mixer_b16, MlpMixer
from g_mlp_pytorch import gMLP, gMLPVision
from models.conv_mlp import ssl_ResMLP, ssl_densenet121

from models.alexnet import AlexNet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

import torch_optimizer
import tensorboard_logger
import cv2
from scipy.stats import wasserstein_distance
from torch.nn.parallel import DistributedDataParallel as DDP


log_dir = './logs'

def set_model(args):
    if args.backbone == 'WideResnet':
        model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
    elif args.backbone == 'ConvMLP':
        model_conv = ssl_densenet121(progress=False, pretrained=args.pre_trained, fusion=args.ConvMlp_fusion)
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, args.n_classes)  # 14 
        
        model_mlp = MlpMixer(
            num_classes=args.n_classes,
            num_blocks=args.num_blocks, #12, [2--9.9M. 4--19.7M]
            patch_size=args.patch_size, #16, 
            hidden_dim=768, 
            tokens_mlp_dim=384, 
            channels_mlp_dim=3072, 
            image_size=112 #224
        )
        if args.pre_trained and args.MLP_ckpt != None:
            model_mlp.load_from(np.load(args.MLP_ckpt))
            print ('Finish loading the MLP pre-trained model!')
        model = [model_conv, model_mlp]
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        save_epoch = checkpoint['epoch']
        print('loaded from checkpoint: %s'%args.checkpoint)
    else:
        save_epoch = 0
        save_optim = None

    if len(args.gpu)>1:
        torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:1001', rank=0, world_size=1)
        model.train()
        model.cuda()
        model = DDP(model, device_ids=[0,1])
    else:
        if args.backbone == 'ConvMLP':
            for i in range(2):
                model[i].train()
                model[i].cuda()
        else:
            model.train()
            model.cuda()

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    if args.eval_ema:
        if args.backbone == 'WideResnet':
            ema_model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
        elif args.backbone == 'ConvMLP':
            ema_model_conv = ssl_densenet121(progress=False, pretrained=args.pre_trained, fusion=args.ConvMlp_fusion)
            num_ftrs = ema_model_conv.classifier.in_features
            ema_model_conv.classifier = nn.Linear(num_ftrs, args.n_classes)  # 14 
            
            ema_model_mlp = MlpMixer(
                num_classes=args.n_classes,
                num_blocks=args.num_blocks, #12, [2--9.9M. 4--19.7M]
                patch_size=args.patch_size, #16, 
                hidden_dim=768, 
                tokens_mlp_dim=384, 
                channels_mlp_dim=3072, 
                image_size=112 #224
            )
            if args.pre_trained and args.MLP_ckpt != None:
                ema_model_mlp.load_from(np.load(args.MLP_ckpt))
                print ('Finish loading the MLP pre-trained model!')
            ema_model = [ema_model_conv, ema_model_mlp]

        if args.backbone == 'ConvMLP':
            for i in range(2):
                if args.dynamic_ema:
                    for param_q, param_k in zip(model[i].parameters(), ema_model[i].parameters()):
                        param_k.data.copy_(param_q.data)  # initialize
                        param_k.requires_grad = True  # not update by gradient for eval_net
                else:
                    for param_q, param_k in zip(model[i].parameters(), ema_model[i].parameters()):
                        param_k.data.copy_(param_q.detach().data)  # initialize
                        param_k.requires_grad = False  # not update by gradient for eval_net
                    ema_model[i].cuda()
                    ema_model[i].eval()
        else:
            if args.dynamic_ema:
                for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                    param_k.data.copy_(param_q.data)  # initialize
                    param_k.requires_grad = True  # not update by gradient for eval_net
                ema_model.cuda()  
                ema_model.train()
            else:
                for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                    param_k.data.copy_(param_q.detach().data)  # initialize
                    param_k.requires_grad = False  # not update by gradient for eval_net
                ema_model.cuda()  
                ema_model.eval() 
    
    else:
        ema_model = None   
    
    if args.dynamic_ema:
        if args.backbone == 'densenet121':
            gate = []
            for i in range(4):
                gate.append(torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).cuda())
            layer_bank = ['denseblock1','denseblock2','denseblock3','denseblock4']
        return model, criteria_x, criteria_u, ema_model, save_epoch, layer_bank, gate
              
    return model, criteria_x, criteria_u, ema_model, save_epoch
    

def sigmoid(x, gamma=1, k=1):
    return 1 / (k + torch.exp(-x*gamma))

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):  # Copy BN params.
        buffer_eval.copy_(buffer_train)  


def calculate_bn_params_diff(net1, net2):
    bn_layers1 = [m for m in net1.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_layers2 = [m for m in net2.modules() if isinstance(m, nn.BatchNorm2d)]
    
    layer_diffs = []
    for bn1, bn2 in zip(bn_layers1, bn_layers2):
        gamma_diff = torch.sum((bn1.weight - bn2.weight) ** 2)
        beta_diff = torch.sum((bn1.bias - bn2.bias) ** 2)
        layer_diff = torch.sqrt(gamma_diff + beta_diff + 1e-6)
        layer_diffs.append(layer_diff.item())
    
    layer_diffs_mean = sum(layer_diffs) / len(layer_diffs)
    return sigmoid(torch.tensor(layer_diffs_mean, device="cuda"))

@torch.no_grad()
def CMH_ema_model_update(model, ema_model, ema_m, ent_p):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    ema_m_update = ema_m * calculate_bn_params_diff(model, ema_model)
    ema_m_update *= sigmoid(ent_p)
    ema_m *= 1 - 0.1*ema_m_update.item()
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):  # Copy BN params.
        buffer_eval.copy_(buffer_train)


@torch.no_grad()
def dynamic_ema_model_update(model, ema_model, ema_m, layer_bank=['denseblock1','denseblock2','denseblock3','denseblock4'], layer_weight=[1,1,1,1]):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for (name_train, param_train), (name_eval, param_eval)  in zip(model.named_parameters(), ema_model.named_parameters()):
        for idx, layer in enumerate(layer_bank):
            if name_train.split('.')[1] == layer:
                layer_weight[idx] = sigmoid(layer_weight[idx]**2)+1.0/0.99-1.0
                _gate = layer_weight[idx]
                param_eval = (param_eval * (ema_m*_gate) + param_train.detach() * (1-(ema_m*_gate))) # param_eval.requires_grad
        if name_train.split('.')[1] not in layer:
            param_eval = (param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):  # Copy BN params. 
        buffer_eval = (buffer_train)  

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def computeAUROC(dataGT, dataPRED, classCount, args):
    classCount = args.n_classes
    if len(dataGT.shape) == 1:
        dataGT = torch.eye(args.n_classes)[dataGT.long().tolist()]

    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC


def mean_class_recall(y_true, y_pred):
    """
    Mean class recall.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    class_recall = []
    target_uniq = np.unique(y_true)

    for label in target_uniq:
        indexes = np.nonzero(label == y_true)[0]
        recall = np.sum(y_true[indexes] == y_pred[indexes]) / len(indexes)
        class_recall.append(recall)
    return np.mean(class_recall)


def compensated_feature(x, y, class_num, max_size=10):
    device = x.device
    compensated_samples = []
    compensated_labels = []
    for i in range(class_num):
        if i not in y:
            continue
        class_samples = x[y == i]
        compensated_num = max_size - (y == i).sum().item()
        mean = class_samples.mean(dim=0).detach().cpu().numpy()
        normed = class_samples - class_samples.mean(dim=0, keepdim=True)

        covariance = (normed.t() @ normed) / (len(class_samples) - 1) + 1e-6 * torch.eye(normed.shape[1], device=device)
        covariance = covariance.detach().cpu().numpy()
        covariance[np.isnan(covariance)] = 1.0

        gaussian_samples = np.random.multivariate_normal(mean, covariance, compensated_num)
        gaussian_samples = torch.from_numpy(gaussian_samples).to(device)
        gaussian_labels = torch.full((compensated_num,), i, dtype=torch.long, device=device)

        compensated_samples.append(gaussian_samples)
        compensated_labels.append(gaussian_labels)

    compensated_samples = torch.cat(compensated_samples, dim=0)
    compensated_labels = torch.cat(compensated_labels, dim=0)

    return compensated_samples, compensated_labels


def train_one_epoch(epoch,
                    model,
                    ema_model,
                    layer_bank, 
                    gate,
                    matchnet,
                    discriminator_img,
                    criteria_x,
                    criteria_u,
                    optim,
                    d_img_optim, 
                    d_match_optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    prob_list,
                    l_probs_list,
                    prob_list_ema,
                    example_stats,
                    ):
    
    if args.backbone == 'ConvMLP':
        model[0].train()
        model[1].train()
    else:
        model.train()
    
    if args.dynamic_ema:
        ema_model.train()
    
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    n_correct_u_lbs_meter = AverageMeter()
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    n_correct_u_lbs_all_meter = AverageMeter()
    guess_recall = AverageMeter()

    ul_true_loss_meter = AverageMeter()
    ul_true_select_meter = AverageMeter()
    ul_true_unselect_meter = AverageMeter()
    
    weight_term_meter = AverageMeter()

    epoch_start = time.time()  # start time

    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters): # Training

        ims_x_weak, lbs_x = next(dl_x)
        
        if args.diff_aug:    
            (ims_u_weak, ims_u_strong, ims_u_strong_mlp), lbs_u_real = next(dl_u)
        else:
            (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)
        
        lbs_x = lbs_x.cuda()
        lbs_u_real = lbs_u_real.cuda()

        bt = ims_x_weak.size(0)
        mu = int(ims_u_weak.size(0) // bt)
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).cuda()

        if args.diff_aug:
            imgs_mlp = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong_mlp], dim=0).cuda()
        
        if 'MLP' not in args.backbone:
            logits, feat = model(imgs,True)
            feat_x = feat[:bt]
            feat_u_w, feat_u_s = torch.split(feat[bt:], bt * mu)
        elif args.backbone == 'ConvMLP':
            if args.ConvMlp_fusion:
                model_conv = model[0]
                model_mlp = model[1]
                logits_conv, side_features_conv = model_conv(imgs)
                logits_mlp, side_features_mlp = model_mlp(imgs)
            else:
                model_conv = model[0]
                model_mlp = model[1]
                logits_conv, feat_conv = model_conv(imgs,True)
                if args.RFC:
                    feat_compensation, lbs_compensation = compensated_feature(torch.split(feat_conv[bt:], bt*mu)[0], lbs_u_real, args.n_classes, max_size=bt*mu)
                    logits_compensation = model_conv.classifier(feat_compensation.float())
                    loss_compensation = criteria_u(logits_compensation, lbs_compensation.long()).mean()

                if args.diff_aug:
                    logits_mlp = model_mlp(imgs_mlp) 
                else:
                    logits_mlp = model_mlp(imgs)         
        else:
            logits = model(imgs)

        if args.backbone == 'ConvMLP':
            logits_x_conv = logits_conv[:bt]
            logits_x_mlp = logits_mlp[:bt]
            logits_u_w, logits_u_s_conv = torch.split(logits_conv[bt:], bt * mu)
            _, logits_u_s_mlp = torch.split(logits_mlp[bt:], bt * mu)
            logits_u_s = logits_u_s_mlp ##
            loss_x = (criteria_x(logits_x_conv, lbs_x.long()) + criteria_x(logits_x_mlp, lbs_x.long())) / 2 # CE loss
        else:
            logits_x = logits[:bt]
            logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu)
            loss_x = criteria_x(logits_x, lbs_x.long()) # CE loss

        if args.mean_teacher:
            logits, feat = ema_model(imgs,True)
            logits_x = logits[:bt]
            logits_u_w, _ = torch.split(logits[bt:], bt * mu)
            loss_x += criteria_x(logits_x, lbs_x.long())
            loss_x /= 2.0

        if args.dynamic_ema:
            probs = torch.softmax(logits_u_w, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)  ## get hard-pseudo label via weak-aug data's pred
            mask = scores.ge(args.thr).float()  ## scores.ge: 1 if scores>=thr; else 0;
        else:
            with torch.no_grad():
                probs = torch.softmax(logits_u_w, dim=1)
                
                if args.DA:
                    prob_list.append(probs.mean(0))
                    if len(prob_list)>32:
                        prob_list.pop(0)   ## maintain a 32-length squeue prob_list to calculate distributions
                    prob_avg = torch.stack(prob_list,dim=0).mean(0)
                    probs = probs / prob_avg
                    if args.standard_DA:
                        l_probs = torch.softmax(logits_x, dim=1)
                        l_probs_list.append(l_probs.mean(0))
                        if len(l_probs_list)>32:
                            l_probs_list.pop(0)   ## maintain a 32-length squeue prob_list to calculate distributions
                        probs = probs * torch.stack(l_probs_list,dim=0).mean(0)
                    probs = probs / probs.sum(dim=1, keepdim=True)   ## Normalize the prob_list       
                    if args.standard_DA:
                        probs = probs ** (1. / args.DA_T) # T=0.5
                        probs = probs / probs.sum(dim=1, keepdim=True)   

                scores, lbs_u_guess = torch.max(probs, dim=1)  ## get hard-pseudo label via weak-aug data's pred

                mask = scores.ge(args.thr).float()  ## scores.ge: 1 if scores>=thr; else 0;


        ul_true_loss = criteria_x(logits_u_w, lbs_u_real.long())
        ul_true_loss_meter.update(ul_true_loss.item())


        ul_true_select_loss = (criteria_u(logits_u_s, lbs_u_real)*mask).sum()/(mask.sum()+1e-5)
        ul_true_select_meter.update(ul_true_select_loss.item())
        ul_true_unselect_loss = (criteria_u(logits_u_s, lbs_u_real)*(1-mask)).sum()/((1-mask).sum()+1e-5)
        ul_true_unselect_meter.update(ul_true_unselect_loss.item())

        if args.dynamic_ema:
            loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()
        else:
            with torch.no_grad():
                probs = probs.detach()
            if args.backbone == 'ConvMLP':
                loss_u = (criteria_u(logits_u_s_conv, lbs_u_guess) * mask).mean()
                loss_u += (criteria_u(logits_u_s_mlp, lbs_u_guess) * mask).mean()
                loss_u /= 2.0

                ######
                # Add Symmetric Loss (option):
                mask1 = scores.ge(0.95).float()  # 0.95,  0.85
                mask2 = scores.ge(0.98).float()  # 0.96,  0.95
                _mask = mask1-mask2  # _mask: scores in [0.95-0.96]

                sym_loss_conv = - ((logits_u_w*logits_u_s_conv).sum(1) / (torch.norm(logits_u_w, dim=1)*torch.norm(logits_u_s_conv, dim=1)+1e-6) * _mask).mean()  # Cosine similarity
                sym_loss_mlp = - ((logits_u_w*logits_u_s_mlp).sum(1) / (torch.norm(logits_u_w, dim=1)*torch.norm(logits_u_s_mlp, dim=1)+1e-6) * _mask).mean()     # Cosine similarity
                loss_u += (sym_loss_conv+sym_loss_mlp)/2.0

                if args.RFC:
                    loss_u += loss_compensation*0.5
                ######

            else:
                loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()  # CE loss of Unlabelled data

        if args.long_tail:
            class_reweight = []
            lbs_u_guess_select = [lbs_u_guess[m] for m,n in enumerate(mask) if n==1]
            for i in range(args.n_classes):
                class_reweight.append(len([q for p,q in enumerate(lbs_u_guess_select) if q==i]))
            reweight_norm = [i/(sum(class_reweight)+1e-5) for i in class_reweight] # + reweight
            loss_u = criteria_u(logits_u_s, lbs_u_guess) * mask
            for i in range(args.n_classes):
                for p,q in enumerate(lbs_u_guess):
                    if q==i and mask[p]==1:
                        loss_u[p] *= reweight_norm[i]
            loss_u = loss_u.mean()

        
        if it == 1 and args.analysis == True:
            logger.info("GT loss: ")
            logger.info(list(((criteria_u(logits_u_s, lbs_u_real)).cpu().detach().numpy())))
            logger.info("Top-1 Prob: ")
            logger.info(list(scores.cpu().detach().numpy()))
            logger.info("Prediction class: ")
            logger.info(list(lbs_u_guess.cpu().detach().numpy()))
            logger.info("Prediction prob: ")
            logger.info(list([probs[i,j].cpu().detach().item() for i,j in enumerate(lbs_u_guess)])[:10])
            logger.info("Mask: ")
            logger.info(list(mask.cpu().detach().numpy()))
            logger.info("Label: ")
            logger.info(list(lbs_u_real.cpu().detach().numpy()))
        
        loss = loss_x + args.lam_u * loss_u 
        
        if args.dynamic_ema:
            for i,j in enumerate((gate)):
                gate[i].register_hook(save_grad('gate'+str(i+1)))

        if args.backbone == 'ConvMLP':
            optim[0].zero_grad()
            optim[1].zero_grad()

            loss.backward()

            nn.utils.clip_grad_value_(model[1].parameters(), clip_value=1.0)  # Only for MLP-mixer
            optim[0].step()
            lr_schdlr[0].step()
            optim[1].step()
            lr_schdlr[1].step()
        else:
            optim.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # Only for MLP-mixer
            optim.step()
            lr_schdlr.step()

        if it % args.ema_step == 0:
            if args.eval_ema:
                    if args.dynamic_ema:
                        dynamic_ema_model_update(model, ema_model, args.ema_m, layer_bank=layer_bank, layer_weight=gate) 
                    else:
                        with torch.no_grad():
                            if args.backbone == 'ConvMLP':
                                if args.CMH:
                                    CMH_ema_model_update(model[0], ema_model[0], args.ema_m, ul_true_loss.item()) 
                                    CMH_ema_model_update(model[1], ema_model[1], args.ema_m, ul_true_loss.item()) 
                                else:
                                    ema_model_update(model[0], ema_model[0], args.ema_m) 
                                    ema_model_update(model[1], ema_model[1], args.ema_m) 
                            else:
                                ema_model_update(model, ema_model, args.ema_m) 

        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        mask_meter.update(mask.mean().item())
        
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        corr_u_lb_all = (lbs_u_guess == lbs_u_real).float()
        n_correct_u_lbs_all_meter.update(corr_u_lb_all.sum().item())
        guess_recall.update((corr_u_lb.sum()/(corr_u_lb_all.sum()+1e-5)).item())

        if (it + 1) % 64 == 0:
            t = time.time() - epoch_start

            if args.backbone == 'ConvMLP':
                lr_log = [pg['lr'] for pg in optim[1].param_groups]
                lr_log = sum(lr_log) / len(lr_log)
            else:
                lr_log = [pg['lr'] for pg in optim.param_groups]
                lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. "
                    "n_correct_u: {:.2f}/{:.2f}. n_correct_u_all: {:.2f}. guess_recall: {:.2f}. Mask:{:.3f}. LR: {:.6f}. Time: {:.2f}".format(
            args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg, loss_x_meter.avg,
            n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, n_correct_u_lbs_all_meter.avg, guess_recall.avg,
            mask_meter.avg, lr_log, t))

            epoch_start = time.time()

    if n_strong_aug_meter.avg == 0:  # For debug
        n_strong_aug_meter.avg += 1

    return loss_x_meter.avg, loss_u_meter.avg, mask_meter.avg, \
            n_correct_u_lbs_meter.avg/n_strong_aug_meter.avg, \
            n_correct_u_lbs_all_meter.avg/lbs_u_real.shape[0], guess_recall.avg, \
            prob_list, l_probs_list


def evaluate(model, ema_model, dataloader, criterion, args=None, each_class_acc=False):
    if args.backbone == 'ConvMLP':
        model[0].eval()
        model[1].eval()
        conv_top1_meter = AverageMeter()
        conv_ema_top1_meter = AverageMeter()
        mlp_top1_meter = AverageMeter()
        mlp_ema_top1_meter = AverageMeter()

        outGT = torch.FloatTensor().cuda()
        conv_outPRED = torch.FloatTensor().cuda()
        mlp_outPRED = torch.FloatTensor().cuda()
    else:
        model.eval()
        top1_meter = AverageMeter()
        ema_top1_meter = AverageMeter()

        mAP_meter = AverageMeter()
        ema_mAP_meter = AverageMeter()
    
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            
            if args.backbone == 'ConvMLP':
                if args.ConvMlp_fusion:
                    logits, side_features_mlp = model(ims)
                else:
                    logits_conv = model[0](ims)
                    logits_mlp = model[1](ims)

                    scores_conv = torch.softmax(logits_conv, dim=1)
                    scores_mlp = torch.softmax(logits_mlp, dim=1)
                    if each_class_acc == True:
                        scores = scores_mlp
                        pred = np.argmax(scores.cpu().numpy(),1)
                        class_acc_list = []
                        for ii in range(10):
                            tmp_label = lbs.cpu().numpy()   # lbs.size=[64], range:(0-9)
                            for iter, kk in enumerate(tmp_label):  # np.argwhere(lbs.cpu().numpy()==ii)
                                if kk != ii:
                                    tmp_label[iter] = -1
                            class_acc_list.append((pred == tmp_label).sum().item() / ((lbs==ii).sum().item()+1e-5))
                    
                    top1_conv, top5_conv = accuracy(scores_conv, lbs, (1, 5))
                    conv_top1_meter.update(top1_conv.item())
                    top1_mlp, top5_mlp = accuracy(scores_mlp, lbs, (1, 5))
                    mlp_top1_meter.update(top1_mlp.item())

                    outGT = torch.cat((outGT, lbs.detach()), 0)
                    conv_outPRED = torch.cat((conv_outPRED, logits_conv.detach()), 0)
                    mlp_outPRED = torch.cat((mlp_outPRED, logits_mlp.detach()), 0)
            else:
                logits = model(ims)
                scores = torch.softmax(logits, dim=1)
                if each_class_acc == True:
                    pred = np.argmax(scores.cpu().numpy(),1)
                    class_acc_list = []
                    for ii in range(10):
                        tmp_label = lbs.cpu().numpy()   # lbs.size=[64], range:(0-9)
                        for iter, kk in enumerate(tmp_label):  # np.argwhere(lbs.cpu().numpy()==ii)
                            if kk != ii:
                                tmp_label[iter] = -1
                        class_acc_list.append((pred == tmp_label).sum().item() / ((lbs==ii).sum().item()+1e-5))
                top1, top5 = accuracy(scores, lbs, (1, 5))
                top1_meter.update(top1.item())
                mAP_score = compute_mAP(scores, lbs)
                mAP_meter.update(mAP_score)
            
            if ema_model is not None:
                if args.backbone == 'ConvMLP':
                    if args.ConvMlp_fusion:
                        logits, side_features_mlp = ema_model(ims)
                    else:
                        logits_conv = ema_model[0](ims)
                        logits_mlp = ema_model[1](ims)
                        scores_conv = torch.softmax(logits_conv, dim=1)
                        scores_mlp = torch.softmax(logits_mlp, dim=1)

                        top1_conv, top5_conv = accuracy(scores_conv, lbs, (1, 5))
                        conv_ema_top1_meter.update(top1_conv.item())
                        top1_mlp, top5_mlp = accuracy(scores_mlp, lbs, (1, 5))
                        mlp_ema_top1_meter.update(top1_mlp.item())
                else:
                    logits = ema_model(ims)
                    scores = torch.softmax(logits, dim=1)
                    top1, top5 = accuracy(scores, lbs, (1, 5))                
                    ema_top1_meter.update(top1.item())

                    ema_mAP_meter.update(mAP_score)
    
    if args.backbone == 'ConvMLP':
        conv_outPRED = torch.argmax(conv_outPRED, dim=1).cpu().data.numpy()
        mlp_outPRED = torch.argmax(mlp_outPRED, dim=1).cpu().data.numpy()
        conv_acc = accuracy_score(outGT.cpu().data.numpy(), conv_outPRED)
        conv_mcr = mean_class_recall(outGT.cpu().data.numpy(), conv_outPRED)
        mlp_acc = accuracy_score(outGT.cpu().data.numpy(), mlp_outPRED)
        mlp_mcr = mean_class_recall(outGT.cpu().data.numpy(), mlp_outPRED)

    if each_class_acc == True:
        if args.backbone == 'ConvMLP':
            return conv_top1_meter.avg, conv_ema_top1_meter.avg, mlp_top1_meter.avg, mlp_ema_top1_meter.avg, class_acc_list, conv_acc, conv_mcr, mlp_acc, mlp_mcr
        else:
            return top1_meter.avg, ema_top1_meter.avg, mAP_meter.avg, ema_mAP_meter.avg, class_acc_list
    else:
        return top1_meter.avg, ema_top1_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--root', default='/home/qsyang2/codes/ssl/datasets', type=str, help='dataset directory')
    
    parser.add_argument('--backbone', type=str, default='WideResnet',   
                        help='name of used backbone')
    parser.add_argument('--wresnet-k', default=2, type=int,  
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int, 
                        help='depth of wide resnet')    
    
    parser.add_argument('--pre-trained', default=True, type=ast.literal_eval, help='use ImageNet pre-trained parameters')
    
    parser.add_argument('--dataset', type=str, default='ISIC', 
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)    

    parser.add_argument('--n-imgs-per-epoch', type=int, default=3560,  #64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=1e-3,  # 0.03
                        help='learning rate for training') 
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--DA', default=True, help='use distribution alignment')

    parser.add_argument('--thr', type=float, default=0.95,   # threshold for hard-pseudo label
                        help='pseudo label threshold')   
    
    parser.add_argument('--exp-dir', default='FixMatch', type=str, help='experiment directory')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')

    parser.add_argument('--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')

    parser.add_argument('--aug-type', type=str, default='RA',
                    help='type of used augmentation techniques')  #['RA', 'CTA']
    
    parser.add_argument('--noisy-rate', default=0, type=float, help='ratio of noisy labels in our CIFAR-10')
    parser.add_argument('--standard-DA', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--DA-T', default=0.5, type=float, help='temperature factor for sharpening of standard_DA')
    parser.add_argument('--long-tail', default=False, type=ast.literal_eval) #bool    
    parser.add_argument('--analysis', default=True, type=ast.literal_eval) #bool
    parser.add_argument('--setting', default='fixmatch_base', type=str, help='setting message for saving logs')

    parser.add_argument('--mean-teacher', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--dynamic-ema', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--ema-step', default=1, type=int, help='step duration to updating ema model')
    
    parser.add_argument('--image-size', type=int, default=112, help='input size')   # [112, 128]
    parser.add_argument('--patch-size', type=int, default=16, help='patch size')
    parser.add_argument('--dim', type=int, default=512, help='dim')
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer') # [sgd, lamb]

    parser.add_argument('--ConvMlp-fusion', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--MLP-ckpt', default='/home/qsyang2/codes/ssl/ours_medical/ckpt/MLP/imagenet21k_Mixer-B_16.npz', type=str) # Pre-train model for MLP

    parser.add_argument('--mlp-lr', type=float, default=1e-3, help='learning rate for training') # 1e-3
    parser.add_argument('--mlp-weight-decay', type=float, default=0.0, help='weight decay')   # 0.0
    parser.add_argument('--num-blocks', type=int, default=4, help='the number of MLP blocks')   # 0.0
    parser.add_argument('--cnn-frozen', type=int, default=0, help='the number of frozen CNN blocks')   # 0.0
    parser.add_argument('--mlp-frozen', type=int, default=0, help='the number of frozen MLP blocks')   # 0.0
    parser.add_argument('--diff-aug', default=False, type=ast.literal_eval) #bool

    parser.add_argument('--RFC', default=False, type=ast.literal_eval) #bool
    parser.add_argument('--CMH', default=False, type=ast.literal_eval) #bool

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    
    logger, output_dir = setup_default_logging(args)

    logger.info(dict(args._get_kwargs()))
    
    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if args.seed > 0:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  
    n_iters_all = n_iters_per_epoch * args.n_epoches  

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    
    layer_bank, gate = None, None
    if args.dynamic_ema:
        model, criteria_x, criteria_u, ema_model, save_epoch, layer_bank, gate = set_model(args)    
    else:
        model, criteria_x, criteria_u, ema_model, save_epoch = set_model(args)
    
    if args.backbone == 'ConvMLP':
        logger.info("Total params: {:.2f}M".format(
            (sum(p.numel() for p in model[0].parameters()) + sum(p.numel() for p in model[1].parameters())) / 1e6))
    else:
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))

    if args.diff_aug:
        dltrain_x, dltrain_u = get_train_loader(
            args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root, method='mlp', aug_type=args.aug_type, seed=args.seed, noisy_rate=args.noisy_rate)
    else:
        dltrain_x, dltrain_u = get_train_loader(
            args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root, method='fixmatch', aug_type=args.aug_type, seed=args.seed, noisy_rate=args.noisy_rate)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root, aug_type=args.aug_type, seed=args.seed)

    if args.backbone == 'ConvMLP':
        conv_wd_params, conv_non_wd_params = [], []
        mlp_wd_params, mlp_non_wd_params = [], []
        
        for name, param in model[0].named_parameters():
            if args.cnn_frozen > 0:
                if name.split('.')[0] == 'block' and int(name.split('.')[1][-1]) < 4-int(args.cnn_frozen) or name.split('.')[0] == 'input_layer':
                    param.requires_grad = False
            else:
                if 'bn' in name:
                    conv_non_wd_params.append(param)  
                else:
                    conv_wd_params.append(param)
        for name, param in model[1].named_parameters():
            if args.mlp_frozen > 0:
                if name.split('.')[0] == 'MixerBlock' and int(name.split('.')[1]) < args.num_blocks-int(args.mlp_frozen) or name == 'stem.weight':
                    param.requires_grad = False
            else:
                if 'bn' in name:
                    mlp_non_wd_params.append(param)  
                else:
                    mlp_wd_params.append(param)
        conv_param_list = [
            {'params': conv_wd_params}, {'params': conv_non_wd_params, 'weight_decay': 0}]
        mlp_param_list = [
            {'params': mlp_wd_params}, {'params': mlp_non_wd_params, 'weight_decay': 0}]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)  
            else:
                wd_params.append(param)
        param_list = [
            {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        optim = torch.optim.SGD(param_list, lr=checkpoint['optim_dict']['param_groups'][0]['lr'])
        optim.load_state_dict(checkpoint['optim_dict']) 
        lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0, last_epoch=n_iters_per_epoch*checkpoint['epoch']) # n_iters_all=1024**2
    else:
        if args.backbone == 'ConvMLP':
            optim_conv = torch.optim.SGD(conv_param_list, lr=args.lr, weight_decay=args.weight_decay, # DenseNet: lr=0.03, decay=5e-4
                            momentum=args.momentum, nesterov=True)
            lr_schdlr_conv = WarmupCosineLrScheduler(optim_conv, n_iters_all, warmup_iter=0)
            optim_mlp = torch.optim.SGD(mlp_param_list, lr=args.mlp_lr, weight_decay=args.mlp_weight_decay, # DenseNet: lr=0.03, decay=5e-4          # For MLP-Mixer
                            momentum=args.momentum, nesterov=True)
            lr_schdlr_mlp = WarmupCosineLrScheduler(optim_mlp, n_iters_all, warmup_iter=0)            # For MLP-Mixer

            optim = [optim_conv, optim_mlp]
            lr_schdlr = [lr_schdlr_conv, lr_schdlr_mlp]
        else:
            optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, # DenseNet: lr=0.03, decay=5e-4
                            momentum=args.momentum, nesterov=True)
            lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)
    
    matchnet, discriminator_img = None, None
    d_img_optim, d_match_optim = None, None

    example_stats = {}
    prob_list = []
    l_probs_list = []
    prob_list_ema = []
    train_args = dict(
        model=model,
        ema_model=ema_model,
        layer_bank=layer_bank, 
        gate=gate,
        matchnet=matchnet,
        discriminator_img=discriminator_img,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        d_img_optim=d_img_optim, 
        d_match_optim=d_match_optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger,
        prob_list=prob_list,
        l_probs_list=l_probs_list,
        prob_list_ema=prob_list_ema,
        example_stats=example_stats
    )
    best_acc = -1
    best_epoch = 0

    global iters
    iters = 0


    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        if args.checkpoint != '':
            epoch = save_epoch

        loss_x, loss_u, mask_mean, guess_label_acc, guess_label_all_acc, guess_label_recall, prob_list, l_probs_list = train_one_epoch(epoch, **train_args)
        
        if (epoch < 200 and epoch % 20 == 0 and epoch != 0) or (epoch > 200 and epoch % 50 == 0):
            if 'MLP' not in args.backbone:
                visualizaion(model, ema_model, dltrain_x, dltrain_u, epoch, output_dir)
        
        if args.backbone == 'ConvMLP':
            conv_top1, conv_ema_top1, mlp_top1, mlp_ema_top1, class_acc, conv_acc, conv_mcr, mlp_acc, mlp_mcr = evaluate(model, ema_model, dlval, criteria_x, args, each_class_acc=True) # MLP
            top1, ema_top1 = mlp_top1, mlp_ema_top1
        else:
            top1, ema_top1, mAP, ema_mAP, class_acc = evaluate(model, ema_model, dlval, criteria_x, args, each_class_acc=True)
    
        tb_logger.log_value('loss_x', loss_x, epoch)
        tb_logger.log_value('loss_u', loss_u, epoch)
        tb_logger.log_value('guess_label_acc', guess_label_acc, epoch)
        tb_logger.log_value('guess_label_all_acc', guess_label_all_acc, epoch)
        tb_logger.log_value('guess_label_recall', guess_label_recall, epoch)
        if args.backbone == 'ConvMLP':
            tb_logger.log_value('conv_test_acc', conv_top1, epoch)
            tb_logger.log_value('conv_test_ema_acc', conv_ema_top1, epoch)
            tb_logger.log_value('mlp_test_acc', mlp_top1, epoch)
            tb_logger.log_value('mlp_test_ema_acc', mlp_ema_top1, epoch)
            tb_logger.log_value('conv_acc', conv_acc, epoch)
            tb_logger.log_value('conv_mcr', conv_mcr, epoch)
            tb_logger.log_value('mlp_acc', mlp_acc, epoch)
            tb_logger.log_value('mlp_mcr', mlp_mcr, epoch)
        else:
            tb_logger.log_value('test_acc', top1, epoch)
            tb_logger.log_value('test_ema_acc', ema_top1, epoch)

            tb_logger.log_value('test_mAP', mAP, epoch)
            tb_logger.log_value('test_ema_mAP', ema_mAP, epoch)

        tb_logger.log_value('mask', mask_mean, epoch)  # len(mask)=ul_bs, 0 or 1 value; mask=1 if p>0.95.
        
        tb_logger.log_value('class1_acc', class_acc[0], epoch)
        tb_logger.log_value('class2_acc', class_acc[1], epoch)
        tb_logger.log_value('class3_acc', class_acc[2], epoch)
        tb_logger.log_value('class4_acc', class_acc[3], epoch)
        tb_logger.log_value('class5_acc', class_acc[4], epoch)
        tb_logger.log_value('class6_acc', class_acc[5], epoch)
        tb_logger.log_value('class7_acc', class_acc[6], epoch)
        tb_logger.log_value('class8_acc', class_acc[7], epoch)
        tb_logger.log_value('class9_acc', class_acc[8], epoch)
        tb_logger.log_value('class10_acc', class_acc[9], epoch)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if (best_acc < top1 and (best_epoch+1) >= 300) or (epoch % 2000 == 0 and 'debug' not in args.setting):
            file_name = os.path.join(output_dir, args.dataset+'_'+str(args.n_labeled)+'_'+str(best_acc)+'_epoch_{}.pth'.format(epoch+1))
            if args.backbone == 'ConvMLP':
                torch.save({
                    'epoch': epoch,
                    'state_dict': model[1].state_dict(),
                    'optim_dict': optim[1].state_dict(),
                    'lr_schdlr': lr_schdlr[1].state_dict(),
                    'prob_list': prob_list,
                    'model': model[1].state_dict(),
                    'ema_model': ema_model[1].state_dict(),
                    },
                    file_name)
            else:
                torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optim.state_dict(),
                'lr_schdlr': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                },
                file_name)

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. mAP: {:.4f}. Ema-mAP: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, mAP, ema_mAP, best_acc, best_epoch))
                

if __name__ == '__main__':
    main()
