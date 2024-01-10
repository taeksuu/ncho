import numpy as np
import torchvision
import torch

import torch.nn.functional as F
import cv2

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def vis_images(batch):

    images = []
    for key in batch:
        img = torchvision.utils.make_grid( batch[key], normalize=True, range=(-1,1), nrow=8).permute(1,2,0).data.cpu().numpy()
        images.append(img)
    return np.concatenate(images, axis=0)
    
def select_dict(dict, keys):
    return {key:dict[key] for key in dict if key in keys}

def mask_dict(dict, mask):

    dict_new = {}
    for key in dict:
        dict_new[key] = dict[key][mask]

    return dict_new

def index_dict(dict, start, end):

    for key in dict:
        dict[key] = dict[key][start:end]

    return dict

def grid_sample_feat(feat_map, x):

    n_batch, n_point, _ = x.shape
    x[...,-1] *= 2

    xy = x[..., :2].transpose(1, 2)
    yz = x[..., 1:].transpose(1, 2)
    xz = x[..., [0, -1]].transpose(1, 2)


    feat_xy = index_custom(feat_map[0], xy).reshape(n_batch, -1, n_point).transpose(1,2)
    feat_yz = index_custom(feat_map[1], yz).reshape(n_batch, -1, n_point).transpose(1,2)
    feat_zx = index_custom(feat_map[2], xz).reshape(n_batch, -1, n_point).transpose(1,2)

    feats = torch.stack([feat_xy, feat_yz, feat_zx], dim=0).sum(axis=0)

    return feats

def expand_cond(cond, x):

    cond = cond[:, None]
    new_shape = list(cond.shape)
    new_shape[0] = x.shape[0]
    new_shape[1] = x.shape[1]
    
    return cond.expand(new_shape)

def rectify_pose(pose, rot):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_rot = cv2.Rodrigues(rot)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = np.linalg.inv(R_abs).dot(R_root)
    new_root = R_rot.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose


def index_custom(feat, uv):
    '''
    Code from https://github.com/shunsukesaito/SCANimate/blob/f2eeb5799fd20fd9d5933472f6aedf1560296cbe/lib/geometry.py

    args:
        feat: (B, C, H, W)
        uv: (B, 2, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, H, W = feat.size()
    _, _, N = uv.size()
    
    x, y = uv[:,0], uv[:,1]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    max_x = W - 1
    max_y = H - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)

    dim2 = W
    dim1 = W * H

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_y0 = base + y0_clamp * dim2
    base_y1 = base + y1_clamp * dim2

    idx_y0_x0 = base_y0 + x0_clamp
    idx_y0_x1 = base_y0 + x1_clamp
    idx_y1_x0 = base_y1 + x0_clamp
    idx_y1_x1 = base_y1 + x1_clamp

    # (B,C,H,W) -> (B,H,W,C)
    im_flat = feat.permute(0,2,3,1).contiguous().view(-1, C)
    i_y0_x0 = torch.gather(im_flat, 0, idx_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_y0_x1 = torch.gather(im_flat, 0, idx_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_y1_x0 = torch.gather(im_flat, 0, idx_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_y1_x1 = torch.gather(im_flat, 0, idx_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_y0_x0 = ((x1 - x) * (y1 - y) * (x1_valid * y1_valid).float()).unsqueeze(1)
    w_y0_x1 = ((x - x0) * (y1 - y) * (x0_valid * y1_valid).float()).unsqueeze(1)
    w_y1_x0 = ((x1 - x) * (y - y0) * (x1_valid * y0_valid).float()).unsqueeze(1)
    w_y1_x1 = ((x - x0) * (y - y0) * (x0_valid * y0_valid).float()).unsqueeze(1)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1 # (B, N, C)

    return output.view(B, N, C).permute(0,2,1).contiguous()
    

def index3d_custom(feat, pts):
    '''
    Code from https://github.com/shunsukesaito/SCANimate/blob/f2eeb5799fd20fd9d5933472f6aedf1560296cbe/lib/geometry.py
    
    args:
        feat: (B, C, D, H, W)
        pts: (B, 3, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, D, H, W = feat.size()
    _, _, N = pts.size()

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)
    z = (D-1.0) * (0.5 * z.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1
    z0 = torch.floor(z).int()
    z1 = z0 + 1

    max_x = W - 1
    max_y = H - 1
    max_z = D - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)
    z0_clamp = torch.clamp(z0, 0, max_z)
    z1_clamp = torch.clamp(z1, 0, max_z)

    dim3 = W
    dim2 = W * H
    dim1 = W * H * D

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_z0_y0 = base + z0_clamp * dim2 + y0_clamp * dim3
    base_z0_y1 = base + z0_clamp * dim2 + y1_clamp * dim3
    base_z1_y0 = base + z1_clamp * dim2 + y0_clamp * dim3
    base_z1_y1 = base + z1_clamp * dim2 + y1_clamp * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clamp
    idx_z0_y0_x1 = base_z0_y0 + x1_clamp
    idx_z0_y1_x0 = base_z0_y1 + x0_clamp
    idx_z0_y1_x1 = base_z0_y1 + x1_clamp
    idx_z1_y0_x0 = base_z1_y0 + x0_clamp
    idx_z1_y0_x1 = base_z1_y0 + x1_clamp
    idx_z1_y1_x0 = base_z1_y1 + x0_clamp
    idx_z1_y1_x1 = base_z1_y1 + x1_clamp

    # (B,C,D,H,W) -> (B,D,H,W,C)
    im_flat = feat.permute(0,2,3,4,1).contiguous().view(-1, C)
    i_z0_y0_x0 = torch.gather(im_flat, 0, idx_z0_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_z0_y0_x1 = torch.gather(im_flat, 0, idx_z0_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_z0_y1_x0 = torch.gather(im_flat, 0, idx_z0_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_z0_y1_x1 = torch.gather(im_flat, 0, idx_z0_y1_x1.unsqueeze(1).expand(-1,C).long())
    i_z1_y0_x0 = torch.gather(im_flat, 0, idx_z1_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_z1_y0_x1 = torch.gather(im_flat, 0, idx_z1_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_z1_y1_x0 = torch.gather(im_flat, 0, idx_z1_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_z1_y1_x1 = torch.gather(im_flat, 0, idx_z1_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)
    z0_valid = (z0 <= max_z) & (z0 >= 0)
    z1_valid = (z1 <= max_z) & (z1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()
    z0 = z0.float()
    z1 = z1.float()

    w_z0_y0_x0 = ((x1 - x) * (y1 - y) * (z1 - z) * (x1_valid * y1_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y0_x1 = ((x - x0) * (y1 - y) * (z1 - z) * (x0_valid * y1_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y1_x0 = ((x1 - x) * (y - y0) * (z1 - z) * (x1_valid * y0_valid * z1_valid).float()).unsqueeze(1)
    w_z0_y1_x1 = ((x - x0) * (y - y0) * (z1 - z) * (x0_valid * y0_valid * z1_valid).float()).unsqueeze(1)
    w_z1_y0_x0 = ((x1 - x) * (y1 - y) * (z - z0) * (x1_valid * y1_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y0_x1 = ((x - x0) * (y1 - y) * (z - z0) * (x0_valid * y1_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y1_x0 = ((x1 - x) * (y - y0) * (z - z0) * (x1_valid * y0_valid * z0_valid).float()).unsqueeze(1)
    w_z1_y1_x1 = ((x - x0) * (y - y0) * (z - z0) * (x0_valid * y0_valid * z0_valid).float()).unsqueeze(1)

    output = w_z0_y0_x0 * i_z0_y0_x0 + w_z0_y0_x1 * i_z0_y0_x1 + w_z0_y1_x0 * i_z0_y1_x0 + w_z0_y1_x1 * i_z0_y1_x1 \
            + w_z1_y0_x0 * i_z1_y0_x0 + w_z1_y0_x1 * i_z1_y0_x1 + w_z1_y1_x0 * i_z1_y1_x0 + w_z1_y1_x1 * i_z1_y1_x1

    return output.view(B, N, C).permute(0,2,1).contiguous()


class Dict2Class(object):
    def __init__(self, dict):
        for key in dict:
            setattr(self, key, dict[key])
  
