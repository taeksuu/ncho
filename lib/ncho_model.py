import os

import hydra
import torch
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.multiprocessing
import torch.autograd as autograd


from lib.model.smpl import SMPLServer
from lib.model.mesh import generate_mesh
from lib.model.sample import PointOnBones
from lib.model.generator import Generator, Generator2D
from lib.model.network import ImplicitNetwork
from lib.model.helpers import expand_cond, vis_images
from lib.utils.render import render_mesh_dict, weights2colors
from lib.model.deformer import skinning
from lib.model.ray_tracing import DepthModule

class BaseModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt
        self.meta_info = meta_info

        self.network_gp_occ = ImplicitNetwork(**opt.network_occ)
        print(self.network_gp_occ)

        self.deformer_gp = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer_gp)

        self.generator_gp = Generator2D(opt.dim_shape, opt.generator.n_layers, opt.generator.init_res)
        print(self.generator_gp)

        self.z_shapes_gp = torch.nn.Embedding(meta_info.n_samples_gp, opt.dim_shape)
        self.z_shapes_gp.weight.data.fill_(0)

        self.network_occ = ImplicitNetwork(**opt.network_occ)
        print(self.network_occ)

        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)

        self.generator = Generator2D(opt.dim_shape, opt.generator.n_layers, opt.generator.init_res)
        print(self.generator)


        self.smpl_server = SMPLServer(gender='neutral')

        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        
        self.z_shapes = torch.nn.Embedding(meta_info.n_samples, opt.dim_shape)
        self.z_shapes.weight.data.fill_(0)
        

        self.network_obj_occ = ImplicitNetwork(**opt.network_obj_occ)
        print(self.network_obj_occ)

        self.generator_obj = Generator2D(opt.dim_shape, opt.generator.n_layers, opt.generator.init_res)
        print(self.generator_obj)

        self.network_both_occ = ImplicitNetwork(**opt.network_both_occ)
        print(self.network_both_occ)

        self.deformer_both = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer_both)

        self.n_samples_only_obj = meta_info.n_samples_obj - meta_info.n_samples_gp
        self.z_shapes_human = torch.nn.Embedding(self.n_samples_only_obj, opt.dim_shape)
        self.z_shapes_human.weight.data.fill_(0)

        self.z_shapes_initial = None
        
        self.z_shapes_obj = torch.nn.Embedding(self.n_samples_only_obj + 1, opt.dim_shape_obj)
        self.z_shapes_obj.weight.data.fill_(0)

        self.data_processor = data_processor


    def configure_optimizers(self):

        grouped_parameters = self.parameters()
        
        def is_included(n): 
            if self.opt.stage =='object':
                if 'z_shapes_human' not in n and 'z_shapes_obj' not in n and 'network_obj' not in n and 'generator_obj' not in n and \
                    'network_both' not in n and 'deformer_both' not in n:
                    return False

            return True

        grouped_parameters = [
            {"params": [p for n, p in list(self.named_parameters()) if is_included(n)], 
            'lr': self.opt.optim.lr, 
            'betas':(0.9,0.999)},
        ]

        optimizer = torch.optim.Adam(grouped_parameters, lr=self.opt.optim.lr)

        if not self.opt.use_gan:
            return optimizer
        else:
            optimizer_d = torch.optim.Adam(self.gan_loss.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d

    def forward(self, pts_d, smpl_tfs, smpl_verts, cond, canonical=False, canonical_shape=False, eval_mode=True, object=False, color=False, fine=False, mask=None, only_near_smpl=False, thrp=False, batch_idx=None):

        n_batch, n_points, n_dim = pts_d.shape
        outputs = {}        
        n_batch, n_points, n_dim = pts_d.shape

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)

        if not mask.any(): 
            return {'occ': -1000*torch.ones( (n_batch, n_points, 1), device=pts_d.device)}

        if canonical_shape:
            pts_c = pts_d 

            occ_pd_obj, feat_pd_obj = self.network_obj_occ(
                                    pts_c, 
                                    cond={'latent': cond['latent_obj']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)


            occ_pd_sp, feat_pd_sp = self.network_occ(
                                    pts_c, 
                                    cond={'latent': cond['latent_sp']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)

            occ_pd_gp, feat_pd_gp = self.network_gp_occ(
                                    pts_c.clone().detach(), 
                                    cond={'latent': cond['latent_gp']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)

            occ_pd_hum = torch.zeros_like(occ_pd_sp)
            occ_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = occ_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
            occ_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = occ_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]

            feat_pd_hum = torch.zeros_like(feat_pd_sp)
            feat_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = feat_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
            feat_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = feat_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]


            input_feat = torch.cat([feat_pd_hum, feat_pd_obj], axis=-1)

            occ_pd_both = self.network_both_occ(
                                    pts_c, 
                                    cond={},
                                    mask=mask,
                                    input_feat=input_feat,
                                    val_pad=-1000,
                                    return_feat=False,
                                    spatial_feat=False,
                                    normalize=True)

        elif canonical:
            pts_c = self.deformer_both.query_cano(pts_d, 
                                    {'betas': cond['betas']}, 
                                    mask=mask)


            occ_pd_obj, feat_pd_obj = self.network_obj_occ(
                                        pts_c, 
                                        cond={'latent': cond['latent_obj']},
                                        mask=mask,
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)
            if not thrp:
                occ_pd_sp, feat_pd_sp = self.network_occ(
                                        pts_c, 
                                        cond={'latent': cond['latent_sp']},
                                        mask=mask,
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

                occ_pd_gp, feat_pd_gp = self.network_gp_occ(
                                        pts_c, 
                                        cond={'latent': cond['latent_gp']},
                                        mask=mask,
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

                if batch_idx:
                    occ_pd_hum = torch.zeros_like(occ_pd_sp)
                    occ_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = occ_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
                    occ_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = occ_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]

                    feat_pd_hum = torch.zeros_like(feat_pd_sp)
                    feat_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = feat_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
                    feat_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = feat_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]
                else:
                    # sdf_pd_hum = sdf_pd_sp
                    occ_pd_hum = occ_pd_sp
                    feat_pd_hum = feat_pd_sp

            else:
                occ_pd_hum, feat_pd_hum = self.network_gp_occ(
                                        pts_c, 
                                        cond={'latent': cond['latent_gp']},
                                        mask=mask,
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)
            input_feat = torch.cat([feat_pd_hum, feat_pd_obj], axis=-1)

            occ_pd_both = self.network_both_occ(
                                    pts_c, 
                                    cond={},
                                    mask=mask,
                                    input_feat=input_feat,
                                    val_pad=-1000,
                                    return_feat=False,
                                    spatial_feat=False,
                                    normalize=True)


        else:             
            pts_c, others = self.deformer_both.forward(pts_d,
                                        {'betas': cond['betas'],
                                        'latent': cond['lbs']},
                                        smpl_tfs,
                                        mask=mask,
                                        eval_mode=eval_mode)

            occ_pd_obj, feat_pd_obj = self.network_obj_occ(
                                        pts_c.reshape((n_batch, -1, n_dim)), 
                                        cond={'latent': cond['latent_obj']},
                                        mask=others['valid_ids'].reshape((n_batch, -1)),
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

            occ_pd_sp, feat_pd_sp = self.network_occ(
                                    pts_c.reshape((n_batch, -1, n_dim)), 
                                    cond={'latent': cond['latent_sp']},
                                    mask=others['valid_ids'].reshape((n_batch, -1)),
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)

            occ_pd_gp, feat_pd_gp = self.network_gp_occ(
                                    pts_c.clone().detach().reshape((n_batch, -1, n_dim)), 
                                    cond={'latent': cond['latent_gp']},
                                    mask=others['valid_ids'].reshape((n_batch, -1)),
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)

            occ_pd_hum = torch.zeros_like(occ_pd_sp)
            occ_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = occ_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
            occ_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = occ_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]

            feat_pd_hum = torch.zeros_like(feat_pd_sp)
            feat_pd_hum[torch.where(batch_idx < self.n_samples_only_obj)] = feat_pd_sp[torch.where(batch_idx < self.n_samples_only_obj)]
            feat_pd_hum[torch.where(batch_idx >= self.n_samples_only_obj)] = feat_pd_gp[torch.where(batch_idx >= self.n_samples_only_obj)]
          
            input_feat = torch.cat([feat_pd_hum, feat_pd_obj], axis=-1)

            occ_pd_both = self.network_both_occ(
                                        pts_c.reshape((n_batch, -1, n_dim)), 
                                        cond={},
                                        mask=others['valid_ids'].reshape((n_batch, -1)),
                                        input_feat=input_feat,
                                        val_pad=-1000,
                                        return_feat=False,
                                        spatial_feat=False,
                                        normalize=True)

            # sdf_pd_both = sdf_pd_both.reshape(n_batch, n_points, -1, 1)
            occ_pd_both = occ_pd_both.reshape(n_batch, n_points, -1, 1)
            occ_pd_both, idx_c_both = occ_pd_both.max(dim=2)

            pts_c = torch.gather(pts_c, 2, idx_c_both.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2)

            occ_pd_hum = occ_pd_hum.reshape(n_batch, n_points, -1, 1)
            occ_pd_hum = torch.gather(occ_pd_hum, 2, idx_c_both.unsqueeze(-1).expand(-1, -1, 1, occ_pd_hum.shape[-1])).squeeze(2)
                    
            occ_pd_obj = occ_pd_obj.reshape(n_batch, n_points, -1, 1)
            occ_pd_obj = torch.gather(occ_pd_obj, 2, idx_c_both.unsqueeze(-1).expand(-1, -1, 1, occ_pd_obj.shape[-1])).squeeze(2)  


        outputs['occ_hum'] = occ_pd_hum       
        outputs['weights_human'] = self.deformer.query_weights(pts_c,  
                                                        cond={
                                                        'betas': cond['betas'],
                                                        'latent': cond['lbs']
                                                        })     
        
        outputs['occ_obj'] = occ_pd_obj
        outputs['occ_both'] = occ_pd_both 
        outputs['pts_c'] = pts_c
        outputs['weights_both'] = self.deformer_both.query_weights(pts_c,  
                                                        cond={
                                                        'betas': cond['betas'],
                                                        'latent': cond['lbs']
                                                        })

        return outputs


    def forward_2d(self, smpl_tfs, smpl_verts, cond, eval_mode=True, fine=True, res=256):

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(smpl_tfs)
        pix_d = pix_d.reshape(1,res*res,2)

        def occ(x, mask=None):

            outputs = self.forward(x, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_smpl=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1] - 0.3, torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1

        d = self.render(pix_d, ray_dirs, occ).detach()
        
        pix_d[...,-1] += d*ray_dirs[...,-1]

        mask = ~d.isinf()

        outputs = self.forward(pix_d, smpl_tfs, smpl_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask)

        outputs['mask'] = mask

        outputs['pts_c'][~mask, :] = 1

        img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        img_mask = np.concatenate([img,mask],axis=-1)

        return img_mask

    def prepare_cond(self, batch):

        cond = {}
        cond['thetas'] =  batch['smpl_params'][:,7:-10]/np.pi
        cond['betas'] = batch['smpl_params'][:,-10:]/10.

        z_shape = batch['z_shape']
        z_shape_obj = batch['z_shape_obj']
        cond['latent_sp'] = self.generator(z_shape)
        cond['latent_gp'] = self.generator_gp(z_shape)
        cond['lbs'] = z_shape
        cond['latent_obj'] = self.generator_obj(z_shape_obj)
        cond['lbs_obj'] = z_shape_obj

        if self.opt.color:
            cond['color'] = batch['z_color']

        return cond
    

    def training_step_coarse(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0

        reg_shape = F.mse_loss(batch['z_shape'], torch.zeros_like(batch['z_shape']))
        self.log('reg_shape', reg_shape)
        loss = loss + self.opt.lambda_reg * reg_shape
        
        reg_lbs = F.mse_loss(cond['lbs'], torch.zeros_like(cond['lbs']))
        self.log('reg_lbs', reg_lbs)
        loss = loss + self.opt.lambda_reg * reg_lbs

        outputs = self.forward(batch['pts_d'], batch['smpl_tfs'],  batch['smpl_verts'], cond, eval_mode=False, only_near_smpl=False)
        loss_bce = F.binary_cross_entropy_with_logits(outputs['occ'], batch['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        
        # Bootstrapping
        num_batch = batch['pts_d'].shape[0]
        if self.current_epoch < self.opt.nepochs_pretrain:

            # Bone occupancy loss
            if self.opt.lambda_bone_occ > 0:
                pts_c, _, occ_gt, _ = self.sampler_bone.get_points(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                outputs = self.forward(pts_c, None, None, cond, canonical=True, only_near_smpl=False)
                loss_bone_occ = F.binary_cross_entropy_with_logits(outputs['occ'], occ_gt.unsqueeze(-1))
                self.log('train_bone_occ', loss_bone_occ)
                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:
                pts_c, w_gt, _ = self.sampler_bone.get_joints(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                w_pd = self.deformer.query_weights(pts_c, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                loss_bone_w = F.mse_loss(w_pd, w_gt)
                self.log('train_bone_w', loss_bone_w)
                loss = loss + self.opt.lambda_bone_w * loss_bone_w

        if self.opt.deformer.nonlinear_offset:
            pts_c, _, occ_gt, _ = self.sampler_bone.get_points(self.smpl_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
            offset = self.deformer.offset(pts_c, cond)
            loss_bone_offset = F.mse_loss(offset, torch.zeros_like(offset))
            self.log('train_bone_offset', loss_bone_offset)
            loss = loss + self.opt.lambda_bone_offset * loss_bone_offset

        # Displacement loss
        pts_c_gt = self.smpl_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1)
        pts_c = self.deformer.query_cano(batch['smpl_verts_cano'], {'betas': cond['betas']})
        loss_disp = F.mse_loss(pts_c, pts_c_gt)

        self.log('train_disp', loss_disp)
        loss = loss + self.opt.lambda_disp * loss_disp

        return loss


    def training_step_color(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0
        
        outputs = self.forward(batch['pts_color'], batch['smpl_tfs'],  batch['smpl_verts'], cond, eval_mode=True, only_near_smpl=False, color=True)
        loss_l1 = F.l1_loss(outputs['verts_color'], batch['color_gt'])
        self.log('color_l1', loss_l1)
        loss = loss + loss_l1

        return loss


    def training_step_object(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0
        num_batch = batch['pts_d'].shape[0]

        reg_shape_human = F.mse_loss(batch['z_shape'], torch.zeros_like(batch['z_shape']))
        self.log('reg_shape_human', reg_shape_human)
        loss = loss + 200 * reg_shape_human

        reg_shape_obj = F.mse_loss(batch['z_shape_obj'][:,self.opt.num_category:], torch.zeros_like(batch['z_shape_obj'][:,self.opt.num_category:]))
        self.log('reg_shape_obj', reg_shape_obj)
        loss = loss + self.opt.lambda_reg * reg_shape_obj
        
        outputs = self.forward(batch['pts_d'], batch['smpl_tfs'],  batch['smpl_verts'], cond, eval_mode=False, only_near_smpl=False, object=True, batch_idx=batch['index'])
        loss_bce = F.binary_cross_entropy_with_logits(outputs['occ_both'], batch['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        loss_bce_obj = F.binary_cross_entropy_with_logits(outputs['occ_obj'], (1 -F.sigmoid(outputs['occ_hum'].detach())) * F.sigmoid(outputs['occ_both'].detach()) * (batch['index'] < self.n_samples_only_obj).float().unsqueeze(1).expand(-1, 2250).unsqueeze(-1))
        self.log('train_bce_obj', loss_bce_obj)
        loss = loss + loss_bce_obj

        loss_bce_pair = F.binary_cross_entropy((F.sigmoid(outputs['occ_hum']) + F.sigmoid(outputs['occ_obj'])).clamp(0,1), batch['occ_gt']) # 014
        self.log('train_bce_pair', loss_bce_pair)
        loss = loss + loss_bce_pair

        self.log('total_loss', loss)
        return loss


    def training_step_fine(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0
        
        outputs = self.forward(batch['cache_pts'], batch['smpl_tfs_img'], None, cond, canonical_shape=True, mask=batch['cache_mask'], fine=True)

        self.gan_loss_input = {
            'norm_real': batch['norm_img'],
            'norm_fake': outputs['norm'].permute(0,2,1).reshape(-1,3,self.opt.img_res,self.opt.img_res)
        }


        if batch_idx%10 == 0 and self.trainer.is_global_zero:
            img = vis_images(self.gan_loss_input)
            self.logger.experiment.log({"imgs":[wandb.Image(img)]})                  
            save_path = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), (255*img).astype(np.uint8)) 

        loss_gan, log_dict = self.gan_loss(self.gan_loss_input, self.global_step, optimizer_idx)
        for key, value in log_dict.items(): self.log(key, value)
        
        loss += self.opt.lambda_gan*loss_gan

        if optimizer_idx == 0:

            if self.opt.norm_loss_3d:           
                outputs = self.forward(batch['pts_surf'], batch['smpl_tfs'],  batch['smpl_verts'], cond, canonical=False, fine=True)
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_surf'])).mean() 
            else:
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_img'].permute(0,2,3,1).flatten(1,2)))[batch['cache_mask']].mean()
        
            self.log('loss_train/train_norm', loss_norm)
            loss += loss_norm

            reg_detail = torch.nn.functional.mse_loss(batch['z_detail'], torch.zeros_like(batch['z_detail']))
            self.log('loss_train/reg_detail', reg_detail)
            loss += self.opt.lambda_reg * reg_detail

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server, load_volume=self.opt.stage!='fine' or self.opt.stage!='color')

        if self.opt.stage == 'coarse':
            batch['z_shape'] = self.z_shapes(batch['index'])
        else:
            batch['z_shape'] = self.z_shapes(batch['index'] * 0)
            batch['z_shape'][torch.where(batch['index'] < self.n_samples_only_obj)] = \
                self.z_shapes_human(batch['index'][torch.where(batch['index'] < self.n_samples_only_obj)])
            batch['z_shape'][torch.where(self.n_samples_only_obj <= batch['index'])] = \
                self.z_shapes_gp(batch['index'][torch.where(self.n_samples_only_obj <= batch['index'])] - self.n_samples_only_obj)

        if self.opt.stage == 'object':
            batch['z_shape_obj'] = self.z_shapes_obj(batch['index'] * 0)
            batch['z_shape_obj'][torch.where(batch['index'] < self.n_samples_only_obj)] = self.z_shapes_obj(batch['index'][torch.where(batch['index'] < self.n_samples_only_obj)])
        else:
            batch['z_shape_obj'] = self.z_shapes_obj(torch.zeros(batch['index'].shape).to(torch.int64).cuda())

        batch['z_shape_obj'][torch.where(batch['index'] >= self.n_samples_only_obj)] = self.z_shapes_obj(torch.ones_like(batch['index'][torch.where(batch['index'] >= self.n_samples_only_obj)]) * self.n_samples_only_obj)
        
        if self.opt.code_category:
            batch['z_shape_obj'] = torch.cat([self.code_category(batch['category'], batch['index']), batch['z_shape_obj']], axis=1)
 
        if self.opt.stage=='coarse':
            loss = self.training_step_coarse(batch, batch_idx)
        elif self.opt.stage == 'object':
            loss = self.training_step_object(batch, batch_idx, optimizer_idx=optimizer_idx)
        elif self.opt.stage == 'color':
            loss = self.training_step_color(batch, batch_idx, optimizer_idx=optimizer_idx)
        elif self.opt.stage == 'fine':
            loss = self.training_step_fine(batch, batch_idx, optimizer_idx=optimizer_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.smpl_server, load_volume=self.opt.stage!='fine' or self.opt.stage!='color')

        if self.opt.stage == 'coarse':
            batch['z_shape'] = self.z_shapes(batch['index'])
        else:
            batch['z_shape'] = self.z_shapes(batch['index'] * 0)
            batch['z_shape'][torch.where(batch['index'] < self.n_samples_only_obj)] = \
                self.z_shapes_human(batch['index'][torch.where(batch['index'] < self.n_samples_only_obj)])
            batch['z_shape'][torch.where(self.n_samples_only_obj <= batch['index'])] = \
                self.z_shapes_gp(batch['index'][torch.where(self.n_samples_only_obj <= batch['index'])] - self.n_samples_only_obj)

        if self.opt.stage == 'object':
            if self.opt.encoding == 'type_location':
                batch['z_shape_obj'] = torch.cat([self.code_type(batch['type'], self.z_shapes_obj(batch['index'])), self.code_location(batch['location'], self.z_shapes_obj(batch['index'])), self.z_shapes_obj(batch['index'])], axis=1)
            elif self.opt.encoding == 'category':
                batch['z_shape_obj'] = torch.cat([self.code_category(batch['category'], self.z_shapes_obj([batch['index']])), self.z_shapes_obj(batch['index'])])
            else:
                batch['z_shape_obj'] = self.z_shapes_obj(batch['index'] * 0)
                batch['z_shape_obj'][torch.where(batch['index'] < self.n_samples_only_obj)] = self.z_shapes_obj(batch['index'][torch.where(batch['index'] < self.n_samples_only_obj)])
        else:
            batch['z_shape_obj'] = self.z_shapes_obj(torch.zeros(batch['index'].shape).to(torch.int64).cuda())

        batch['z_shape_obj'][torch.where(batch['index'] >= self.n_samples_only_obj)] = self.z_shapes_obj(torch.ones_like(batch['index'][torch.where(batch['index'] >= self.n_samples_only_obj)]) * self.n_samples_only_obj)

        if self.opt.code_category:
            batch['z_shape_obj'] = torch.cat([self.code_category(batch['category'], batch['index']), batch['z_shape_obj']], axis=1)

        if batch_idx == 0 and self.trainer.is_global_zero:
            with torch.no_grad(): self.plot(batch)   

    def extract_mesh_neural(self, smpl_verts, smpl_tfs, cond, res_up=3, thrp=False):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False, object=self.opt.stage=='object', thrp=thrp)
            return outputs['occ_both'].reshape(-1,1)


        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up, clean=False)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, object=self.opt.stage=='object', color=self.opt.stage=='color', fine=self.opt.stage=='fine', only_near_smpl=False, thrp=thrp)
        
        mesh['weights'] = outputs['weights_human'][0].detach()#.clamp(0,1)[0]
        mesh['weights'][(outputs['occ_hum'][0] > 0).squeeze(-1)] = outputs['weights_both'][0][(outputs['occ_hum'][0] > 0).squeeze(-1)].detach()

        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        if self.opt.stage == 'color':
            mesh['verts_color'] = outputs['verts_color'][0].detach()
            mesh['verts_color'] = 0.5 * mesh['verts_color'] + 0.5
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh

    def extract_mesh_naive(self, smpl_verts, smpl_tfs, cond, res_up=3, thrp=False):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False, object=self.opt.stage=='object', thrp=thrp)
            return torch.max(outputs['occ_hum'].reshape(-1,1), outputs['occ_obj'].reshape(-1,1))


        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up, clean=False)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, object=self.opt.stage=='object', color=self.opt.stage=='color', fine=self.opt.stage=='fine', only_near_smpl=False, thrp=thrp)
        
        mesh['weights'] = outputs['weights_both'][0].detach()#.clamp(0,1)[0]

        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        if self.opt.stage == 'color':
            mesh['verts_color'] = outputs['verts_color'][0].detach()
            mesh['verts_color'] = 0.5 * mesh['verts_color'] + 0.5
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh

    def extract_mesh_mixed(self, smpl_verts, smpl_tfs, cond, res_up=3, thrp=False):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False, object=self.opt.stage=='object', thrp=thrp)
            if self.opt.stage == 'object':
                return 0.5 * torch.max(outputs['occ_hum'].reshape(-1,1), outputs['occ_obj'].reshape(-1,1)) + 0.5 * outputs['occ_both'].reshape(-1,1)
            else:
                return F.sigmoid(outputs['occ'].reshape(-1,1))

        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up, clean=False)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, object=self.opt.stage=='object', color=self.opt.stage=='color', fine=self.opt.stage=='fine', only_near_smpl=False, thrp=thrp)
        
        mesh['weights'] = outputs['weights_both'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        if self.opt.stage == 'color':
            mesh['verts_color'] = outputs['verts_color'][0].detach()
            mesh['verts_color'] = 0.5 * mesh['verts_color'] + 0.5
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh

    def extract_obj_mesh(self, smpl_verts, smpl_tfs, cond, res_up=3, thrp=False):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False, object=self.opt.stage=='object', thrp=thrp)
            return outputs['occ_obj'].reshape(-1,1)

        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up, level_set=0, clean=False)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, object=self.opt.stage=='object', color=self.opt.stage=='color', fine=self.opt.stage=='fine', only_near_smpl=False, thrp=thrp)
        
        mesh['weights'] = outputs['weights_both'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        if self.opt.stage == 'color':
            mesh['verts_color'] = outputs['verts_color'][0].detach()
            mesh['verts_color'] = 0.5 * mesh['verts_color'] + 0.5
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

            return mesh

    def extract_human_mesh(self, smpl_verts, smpl_tfs, cond, res_up=3, thrp=False):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, smpl_tfs, smpl_verts, cond, canonical=True, only_near_smpl=False, object=self.opt.stage=='object', thrp=thrp)
            return outputs['occ_hum'].reshape(-1,1)

        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up, clean=False)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(smpl_verts), 
                'faces': torch.tensor(mesh.faces, device=smpl_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, smpl_tfs, smpl_verts, cond, canonical=True, object=self.opt.stage=='object', color=self.opt.stage=='color', fine=self.opt.stage=='fine', only_near_smpl=False, thrp=thrp)
        
        mesh['weights'] = outputs['weights_human'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=smpl_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        if self.opt.stage == 'color':
            mesh['verts_color'] = outputs['verts_color'][0].detach()
            mesh['verts_color'] = 0.5 * mesh['verts_color'] + 0.5
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['norm'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh
        

    def deform_mesh(self, mesh, smpl_tfs, cond):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        mesh = copy.deepcopy(mesh)

        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)

        if self.opt.deformer.nonlinear_offset:
            _cond = { key:expand_cond(cond[key], mesh['verts'].unsqueeze(0)) for key in cond}
            _cond = { key:_cond[key][0] for key in _cond}
            _cond['latent_obj'] = _cond['lbs_obj']

            mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs, cond, opt=self.opt.deformer, offset=self.deformer.offset(mesh['verts'], _cond))

        else:
            mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)

        
        if 'norm' in mesh:
            mesh['norm']  = skinning( mesh['norm'], mesh['weights'], smpl_tfs, normal=True, opt=self.opt.deformer)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh

    def deform_human_mesh(self, mesh, smpl_tfs, cond):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        mesh = copy.deepcopy(mesh)

        smpl_tfs = smpl_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)

        if self.opt.deformer.nonlinear_offset:
            _cond = { key:expand_cond(cond[key], mesh['verts'].unsqueeze(0)) for key in cond}
            _cond = { key:_cond[key][0] for key in _cond}
            _cond['latent_obj'] = _cond['lbs_obj']

            mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs, cond, opt=self.opt.deformer, offset=self.deformer.offset(mesh['verts'], _cond))

        else:
            mesh['verts'] = skinning(mesh['verts'], mesh['weights'], smpl_tfs)

        
        if 'norm' in mesh:
            mesh['norm']  = skinning( mesh['norm'], mesh['weights'], smpl_tfs, normal=True, opt=self.opt.deformer)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh

    def render_one(self, batch, hum_id, obj_id, obj_list, dataset_path, type='naive', thrp=True):

        scan_info = obj_list.iloc[hum_id]
        obj_info = obj_list.iloc[obj_id]
        f = np.load(os.path.join(dataset_path, scan_info['id'], 'sdf.npz') )

        batch['smpl_params'] = f['smpl_params'].astype(np.float32)
        smpl_output = self.smpl_server(torch.tensor(batch['smpl_params']).to(device=batch['z_shape'].device)[None], absolute=False)
        batch.update(smpl_output)

        cond = {}
        cond['thetas'] = torch.tensor(batch['smpl_params'][4:76]/np.pi).to(device=batch['z_shape'].device)[None]
        cond['betas'] = torch.tensor(batch['smpl_params'][76:]/10.).to(device=batch['z_shape'].device)[None]


        if hum_id < self.n_samples_only_obj:
            z_shape = self.z_shapes_human(torch.tensor([hum_id]).to(device=batch['z_shape'].device))
        else:
            z_shape = self.z_shapes_gp(torch.tensor([hum_id]).to(device=batch['z_shape'].device) - self.n_samples_only_obj)

        z_shape_obj = self.z_shapes_obj(torch.tensor([obj_id]).to(device=batch['z_shape'].device))
        if self.opt.code_category:
            z_shape_obj = torch.cat([self.code_category(torch.tensor([int(obj_info['category'])]).to(device=batch['z_shape'].device), batch['index']), z_shape_obj], axis=1)

        if thrp:
            cond['latent_sp'] = self.generator_gp(z_shape)
            cond['latent_gp'] = self.generator_gp(z_shape)
        else:
            cond['latent_sp'] = self.generator(z_shape)
            cond['latent_gp'] = self.generator(z_shape)

        cond['lbs'] = z_shape

        cond['latent_obj'] = self.generator_obj(z_shape_obj)
        cond['lbs_obj'] = z_shape_obj

        if type == 'naive':
            surf_pred_cano = self.extract_mesh_naive(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3, thrp=thrp)
        elif type == 'human':
            surf_pred_cano = self.extract_human_mesh(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3, thrp=thrp)
        elif type == 'object':
            surf_pred_cano = self.extract_obj_mesh(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3, thrp=thrp)
        elif type == 'neural':
            surf_pred_cano = self.extract_mesh_neural(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3, thrp=thrp)
        elif type == 'mixed':
            surf_pred_cano = self.extract_mesh_mixed(batch['smpl_verts_cano'], batch['smpl_tfs'], cond, res_up=3, thrp=thrp)

        if type == 'human':
            surf_pred_def  = self.deform_human_mesh(surf_pred_cano, batch['smpl_tfs'], cond)
        else:
            surf_pred_def  = self.deform_mesh(surf_pred_cano, batch['smpl_tfs'], cond)

        return render_mesh_dict(surf_pred_def, mode='n')


    def plot(self, batch):

        import pandas
        obj_list = pandas.read_csv(hydra.utils.to_absolute_path('./lib/dataset/ts_200_obj_cat_part.csv'),dtype=str)
        dataset_path = self.meta_info.dataset_path

        temp = batch.copy()

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]


        img_list = [] # backpack 1
        img_list.append(self.render_one(batch, 48, 48, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 48, 48, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 48, 48, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 349, 48, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 349, 48, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 349, 48, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 362, 48, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 362, 48, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 362, 48, obj_list, dataset_path, type='neural'))

        img_all_1 = np.concatenate(img_list, axis=1)
        
        img_list = [] # outer
        img_list.append(self.render_one(batch, 146, 146, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 146, 146, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 146, 146, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 342, 146, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 342, 146, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 342, 146, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 796, 146, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 796, 146, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 796, 146, obj_list, dataset_path, type='neural'))

        img_all_2 = np.concatenate(img_list, axis=1)

        img_list = [] # outer
        img_list.append(self.render_one(batch, 156, 156, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 156, 156, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 156, 156, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 342, 156, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 342, 156, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 342, 156, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 796, 156, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 796, 156, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 796, 156, obj_list, dataset_path, type='neural'))

        img_all_3 = np.concatenate(img_list, axis=1)

        
        img_list = [] # scarf
        img_list.append(self.render_one(batch, 171, 171, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 171, 171, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 171, 171, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 475, 171, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 475, 171, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 475, 171, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 436, 171, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 436, 171, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 436, 171, obj_list, dataset_path, type='neural'))

        img_all_4 = np.concatenate(img_list, axis=1)

        img_list = [] # scarf
        img_list.append(self.render_one(batch, 256, 256, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 256, 256, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 256, 256, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 475, 256, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 475, 256, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 475, 256, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 436, 256, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 436, 256, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 436, 256, obj_list, dataset_path, type='neural'))

        img_all_5 = np.concatenate(img_list, axis=1)

        img_list = [] # hat
        img_list.append(self.render_one(batch, 272, 272, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 272, 272, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 272, 272, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 571, 272, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 571, 272, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 571, 272, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 772, 272, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 772, 272, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 772, 272, obj_list, dataset_path, type='neural'))
    
        img_all_6 = np.concatenate(img_list, axis=1)

        img_list = [] # hat
        img_list.append(self.render_one(batch, 318, 318, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 318, 318, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 318, 318, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 571, 318, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 571, 318, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 571, 318, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 772, 318, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 772, 318, obj_list, dataset_path, type='mixed'))
        img_list.append(self.render_one(batch, 772, 318, obj_list, dataset_path, type='neural'))
    
        img_all_7 = np.concatenate(img_list, axis=1)


        img_all = np.concatenate([img_all_1,img_all_2,img_all_3,img_all_4,img_all_5,img_all_6,img_all_7], axis=0)
        
        self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all)        


    def plot_scarf(self, batch):

        import pandas
        obj_list = pandas.read_csv(hydra.utils.to_absolute_path('./lib/dataset/ts_200_scarf.csv'),dtype=str)
        dataset_path = self.meta_info.dataset_path

        temp = batch.copy()

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        img_list = [] # scarf 1
        img_list.append(self.render_one(batch, 190, 190, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 190, 190, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 190, 190, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 506, 190, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 506, 190, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 467, 190, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 467, 190, obj_list, dataset_path, type='neural'))

        img_all_7 = np.concatenate(img_list, axis=1)

        img_list = [] # scarf 2
        img_list.append(self.render_one(batch, 330, 330, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 330, 330, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 330, 330, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 506, 330, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 506, 330, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 467, 330, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 467, 330, obj_list, dataset_path, type='neural'))

        img_all_8 = np.concatenate(img_list, axis=1)

        img_list = [] # scarf 3
        img_list.append(self.render_one(batch, 334, 334, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 334, 334, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 334, 334, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 506, 334, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 506, 334, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 467, 334, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 467, 334, obj_list, dataset_path, type='neural'))

        img_all_9 = np.concatenate(img_list, axis=1)

        img_all = np.concatenate([img_all_7,img_all_8,img_all_9])

        self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all)    


    def plot_bag(self, batch):

        import pandas
        obj_list = pandas.read_csv(hydra.utils.to_absolute_path('./lib/dataset/ts_200_bag.csv'),dtype=str)
        dataset_path = self.meta_info.dataset_path

        temp = batch.copy()

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        img_list = [] # backpack 1
        img_list.append(self.render_one(batch, 80, 80, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 80, 80, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 80, 80, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 307, 80, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 307, 80, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 320, 80, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 320, 80, obj_list, dataset_path, type='neural'))

        img_all_1 = np.concatenate(img_list, axis=1)

        img_list = [] # backpack 2
        img_list.append(self.render_one(batch, 205, 205, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 205, 205, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 205, 205, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 307, 205, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 307, 205, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 320, 205, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 320, 205, obj_list, dataset_path, type='neural'))

        img_all_2 = np.concatenate(img_list, axis=1)

        img_list = [] # backpack 3
        img_list.append(self.render_one(batch, 280, 280, obj_list, dataset_path, type='human', thrp=False))
        img_list.append(self.render_one(batch, 280, 280, obj_list, dataset_path, type='object', thrp=False))
        img_list.append(self.render_one(batch, 280, 280, obj_list, dataset_path, type='neural', thrp=False))

        img_list.append(self.render_one(batch, 307, 280, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 307, 280, obj_list, dataset_path, type='neural'))

        img_list.append(self.render_one(batch, 320, 280, obj_list, dataset_path, type='naive'))
        img_list.append(self.render_one(batch, 320, 280, obj_list, dataset_path, type='neural'))

        img_all_3 = np.concatenate(img_list, axis=1)
        img_all = np.concatenate([img_all_1,img_all_2,img_all_3])

        self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all)        

    def sample_codes(self, n_sample, std_scale=1):
        device = self.z_shapes.weight.device

        mean_shapes = self.z_shapes_gp.weight.data.mean(0)
        std_shapes = self.z_shapes_gp.weight.data.std(0)
        
        mean_shapes_obj = self.z_shapes_obj.weight.data.mean(0)
        std_shapes_obj = self.z_shapes_obj.weight.data.std(0)

        z_shape = torch.randn(n_sample, self.opt.dim_shape, device=device)
        z_shape_obj = torch.randn(n_sample, self.opt.dim_shape_obj, device=device)  

        z_shape = z_shape*std_shapes*std_scale+mean_shapes
        z_shape_obj = z_shape_obj*std_shapes_obj*std_scale+mean_shapes_obj

        return z_shape, z_shape_obj

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if k not in state_dict:
                    hello = 1
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def initialize_z_shape_human(self):
        self.z_shapes_human.weight.data = self.z_shapes.weight.data.mean(axis=0).repeat(self.n_samples_only_obj, 1)
        self.z_shapes_initial = self.z_shapes.weight.data.mean(axis=0)

        
    def code_type(self, type, dim):
        code = torch.zeros((dim.shape[0], self.opt.dim_type))
        code[torch.arange(type.size(dim=0)), type] = 1
        return code.cuda()

    def code_location(self, location, dim):
        code = torch.zeros((dim.shape[0], self.opt.dim_location))
        code[torch.arange(location.size(dim=0)), location] = 1
        return code.cuda()

    def code_category(self, category, dim):
        code = torch.zeros((dim.shape[0], self.opt.num_category))
        code[torch.arange(category.size(dim=0)), category] = 1
        return code.cuda()
