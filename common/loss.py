import torch
import numpy as np


def mpjpe(output_3D, out_target):
	loss = torch.mean(torch.norm(output_3D - out_target, dim=-1))

	return loss


def weighted_mpjpe(predicted, target, w_mpjpe):
    return torch.mean(w_mpjpe * torch.norm(predicted - target, dim=len(target.shape)-1))


def temporal_consistency(predicted, target, w_mpjpe):
    dif_seq = predicted[:,1:,:,:] - predicted[:,:-1,:,:]
    weights_joints = torch.ones_like(dif_seq).cuda()
    weights_joints = torch.mul(weights_joints.permute(0,1,3,2), w_mpjpe).permute(0,1,3,2)
    dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

    return dif_seq


def mean_velocity(predicted, target, axis=0):
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))





