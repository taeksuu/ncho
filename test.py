
import glob
import os
import pickle

import hydra
import imageio
import numpy as np
import open3d as o3d
import pandas
import pytorch_lightning as pl
import torch
from pytorch3d.io import save_obj
from tqdm import tqdm

from lib.dataset.datamodule import DataProcessor
from lib.ncho_model import BaseModel
from lib.model.helpers import Dict2Class
from lib.utils.render import Renderer, render_mesh_dict


@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(opt.seed, workers=True)
    torch.set_num_threads(10) 

    scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.datamodule.data_list))
    scan_info_obj = pandas.read_csv(hydra.utils.to_absolute_path(opt.datamodule.data_list_obj))
    scan_info_thrp = pandas.read_csv(hydra.utils.to_absolute_path(opt.datamodule.data_list_thrp))
    meta_info = Dict2Class({'n_samples': len(scan_info),
                            'n_samples_obj': len(scan_info_obj),
                            'n_samples_gp': len(scan_info_thrp)})

    data_processor = DataProcessor(opt.datamodule)
    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()

    model.opt.stage=='object'
    smpl_param_zero = torch.zeros((1,86)).cuda().float()
    smpl_param_zero[:,0] = 1

    batch_list = []
   

    if opt.eval_mode == 'sample':
        z_shapes, _ = model.sample_codes(5)

        for i in range(len(z_shapes)):
            pickle_path = '../../data/sample_smpl/00000_smpl.pkl'
            file = pandas.read_pickle(pickle_path)
            smpl_param = smpl_param_zero.clone()
            smpl_param_file = np.concatenate([np.ones((1,1)), 
                                    np.zeros( (1,3)),
                                    file['global_orient'].reshape(1,-1),
                                    file['body_pose'].reshape(1,-1)[:,3:],
                                    file['betas'][:,:10]], axis=1)[0]
            smpl_param[:, :] = torch.from_numpy(smpl_param_file)

            batch = {'z_shape': z_shapes[i][None],
                    'z_shape_obj': model.z_shapes_obj.weight.data[0][None],
                    'smpl_params': smpl_param,
                    }
            
            batch_list.append(batch)

    if opt.eval_mode == 'interp_hum':
        train_data_list = pandas.read_csv('../../' + opt.datamodule['data_list'][2:],dtype=str)
        train_data_list_obj = pandas.read_csv('../../' + opt.datamodule['data_list_obj'][2:],dtype=str)
        thrp_list = pandas.read_csv('../../' + opt.datamodule['data_list_thrp'][2:],dtype=str)

        idx_a = 16 # human shapes manually set for larger difference between samples to be interpolated, can be randomized
        idx_b = 7

        target_obj = [0]
        for idx_c in target_obj:
            z_shape_a = model.z_shapes_gp.weight.data[idx_a]
            z_shape_b = model.z_shapes_gp.weight.data[idx_b]
            z_shape_obj = model.z_shapes_obj.weight.data[idx_c]

            pickle_path = '../../data/sample_smpl/00000_smpl.pkl'
            file = pandas.read_pickle(pickle_path)
            smpl_param = smpl_param_zero.clone()
            smpl_param_file = np.concatenate([np.ones( (1,1)), 
                                    np.zeros((1,3)),
                                    file['body_pose'].reshape(1,-1),
                                    file['betas'][:,:10]], axis=1)[0]
            smpl_param[:, :] = torch.from_numpy(smpl_param_file)

            for i in range(5):
                z_shape = torch.lerp(z_shape_a, z_shape_b, i/5)
                batch = {'z_shape': z_shape[None],
                        'z_shape_obj': z_shape_obj[None],
                        'smpl_params': smpl_param}
                batch_list.append(batch)


    
    if opt.eval_mode == 'interp_obj':
        train_data_list = pandas.read_csv('../../' + opt.datamodule['data_list'][2:],dtype=str)
        train_data_list_obj = pandas.read_csv('../../' + opt.datamodule['data_list_obj'][2:],dtype=str)
        thrp_list = pandas.read_csv('../../' + opt.datamodule['data_list_thrp'][2:],dtype=str)

        idx_a = 70 # first sample for each category of backpacks
        idx_b = 188
        
        target = [0]
        for idx_c in target:
            z_shape = model.z_shapes_gp.weight.data[idx_c]
            z_shape_obj_a = model.z_shapes_obj.weight.data[idx_a]
            z_shape_obj_b = model.z_shapes_obj.weight.data[idx_b]

            pickle_path = '../../data/sample_smpl/00000_smpl.pkl'
            file = pandas.read_pickle(pickle_path)
            smpl_param = smpl_param_zero.clone()
            smpl_param_file = np.concatenate([np.ones( (1,1)), 
                                    np.zeros((1,3)),
                                    file['body_pose'].reshape(1,-1),
                                    file['betas'][:,:10]], axis=1)[0]
            smpl_param[:, :] = torch.from_numpy(smpl_param_file)

            for i in range(5):
                z_shape_obj = torch.lerp(z_shape_obj_a, z_shape_obj_b, i/5)
                batch = {'z_shape': z_shape[None],
                        'z_shape_obj': z_shape_obj[None],
                        'smpl_params': smpl_param}
                batch_list.append(batch)


    if opt.eval_mode == 'dis_hum':
        names = []
        train_data_list = pandas.read_csv('../../' + opt.datamodule['data_list'][2:],dtype=str)
        train_data_list_obj = pandas.read_csv('../../' + opt.datamodule['data_list_obj'][2:],dtype=str)
        thrp_list = pandas.read_csv('../../' + opt.datamodule['data_list_thrp'][2:],dtype=str)
        target = np.random.choice(range(len(thrp_list)), 1)
        target_obj = [0, 70, 128, 188, 246] # first sample for each category of backpacks

        for i in target:
            pickle_path = '../../data/sample_smpl/00000_smpl.pkl'
            file = pandas.read_pickle(pickle_path)
            smpl_param = smpl_param_zero.clone()
            smpl_param_file = np.concatenate([np.ones( (1,1)), 
                                    np.zeros((1,3)),
                                    file['body_pose'].reshape(1,-1),
                                    file['betas'][:,:10]], axis=1)[0]
            smpl_param[:, :] = torch.from_numpy(smpl_param_file)

            for j in target_obj:
                names.append((thrp_list.iloc[i]['id'] + "_" + train_data_list_obj.iloc[j]['id']).replace('/', '-'))
                cat = int(train_data_list_obj.iloc[j]['category'])
                batch = {'z_shape':  model.z_shapes_gp.weight.data[i][None],
                        'z_shape_obj': model.z_shapes_obj.weight.data[j][None],
                        'smpl_params': smpl_param,
                        }
                batch_list.append(batch)

    if opt.eval_mode == 'dis_obj':
        names = []
        train_data_list = pandas.read_csv('../../' + opt.datamodule['data_list'][2:],dtype=str)
        train_data_list_obj = pandas.read_csv('../../' + opt.datamodule['data_list_obj'][2:],dtype=str)
        thrp_list = pandas.read_csv('../../' + opt.datamodule['data_list_thrp'][2:],dtype=str)
        target = np.random.choice(range(len(thrp_list)), 5)
        target_obj = [0]

        for i in target:
            pickle_path = '../../data/sample_smpl/00000_smpl.pkl'
            file = pandas.read_pickle(pickle_path)
            smpl_param = smpl_param_zero.clone()
            smpl_param_file = np.concatenate([np.ones( (1,1)), 
                                    np.zeros((1,3)),
                                    file['body_pose'].reshape(1,-1),
                                    file['betas'][:,:10]], axis=1)[0]
            smpl_param[:, :] = torch.from_numpy(smpl_param_file)

            for j in target_obj:
                names.append((thrp_list.iloc[i]['id'] + "_" + train_data_list_obj.iloc[j]['id']).replace('/', '-'))
                cat = int(train_data_list_obj.iloc[j]['category'])
                batch = {'z_shape':  model.z_shapes_gp.weight.data[i][None],
                        'z_shape_obj': model.z_shapes_obj.weight.data[j][None],
                        'smpl_params': smpl_param,
                        }
                batch_list.append(batch)

    
    output_folder = opt.eval_mode
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(batch_list)):
            cond = model.prepare_cond(batch)
            batch_smpl = data_processor.process_smpl({'smpl_params': batch['smpl_params']}, model.smpl_server)

            if 'dis_' in opt.eval_mode:
                mesh_cano = model.extract_mesh_neural(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond, res_up=4, thrp=True)
                mesh_def = model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'], cond)
                o3d_mesh_def = o3d.geometry.TriangleMesh()
                o3d_mesh_def.vertices = o3d.utility.Vector3dVector(mesh_def['verts'].cpu().detach().numpy())
                o3d_mesh_def.triangles = o3d.utility.Vector3iVector(mesh_def['faces'].cpu().detach().numpy())
                o3d.io.write_triangle_mesh(os.path.join(output_folder, f'{names[i][4:]}_neural.obj'), o3d_mesh_def, write_ascii=True)
            else:
                mesh_cano = model.extract_mesh_neural(batch_smpl['smpl_verts_cano'], batch_smpl['smpl_tfs'], cond, res_up=4, thrp=True)
                mesh_def = model.deform_mesh(mesh_cano, batch_smpl['smpl_tfs'], cond)
                o3d_mesh_def = o3d.geometry.TriangleMesh()
                o3d_mesh_def.vertices = o3d.utility.Vector3dVector(mesh_def['verts'].cpu().detach().numpy())
                o3d_mesh_def.triangles = o3d.utility.Vector3iVector(mesh_def['faces'].cpu().detach().numpy())
                o3d.io.write_triangle_mesh(os.path.join(output_folder, f'{i:03d}_neural.obj'), o3d_mesh_def)


if __name__ == '__main__':
    main()