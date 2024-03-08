import os
import sys
import time
import glob
import time
import torch
import random
import socket
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from common.utils import *
import torch.optim as optim
from common.camera import *
import common.loss as eval_loss
from common.arguments import parse_args
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

exec('from model.' + args.model + ' import Model')

def train(dataloader, model, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        output_3D = model(input_2D)

        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0

        w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()

        loss_w_mpjpe = eval_loss.weighted_mpjpe(output_3D, out_target, w_mpjpe)
        loss_temporal = eval_loss.temporal_consistency(output_3D, out_target, w_mpjpe)
        loss_mean_velocity = eval_loss.mean_velocity(output_3D, out_target, axis=1)

        loss = loss_w_mpjpe + 0.5 * loss_temporal + 2.0 * loss_mean_velocity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, dataloader, model):
    model.eval()

    action_error = define_error_list(actions)

    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]

    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        if args.stride == 1:
            out_target = out_target[:, args.pad].unsqueeze(1)
            output_3D = output_3D[:, args.pad].unsqueeze(1)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)

    p1, p2 = print_error(args.dataset, action_error, 1)

    return p1, p2

if __name__ == '__main__':
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    if args.train:
        train_data = Fusion(args, dataset, args.root_path, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
    test_data = Fusion(args, dataset, args.root_path, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = Model(args).cuda()

    if args.previous_dir != '':
        Load_model(args, model)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        if args.train: 
            loss = train(train_dataloader, model, optimizer, epoch)
            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            p1, p2 = test(actions, test_dataloader, model)
            mpjpes.append(p1)

        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')
            args.previous_best = p1

        if args.train:
            logging.info('epoch: %d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, args.previous_best))
            print('%d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, args.previous_best))
        
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
        else:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break

