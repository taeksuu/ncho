import os
import PIL
import torch
import hydra
import pandas
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
import torchvision.transforms as transforms

from lib.model.helpers import rectify_pose, Dict2Class

class DataSet(torch.utils.data.Dataset):

    def __init__(self, dataset_path, val=False, opt=None, type=None):

        self.dataset_path = hydra.utils.to_absolute_path(dataset_path)

        self.cache_path = hydra.utils.to_absolute_path(opt.cache_path)
        self.cache_path = os.path.join(os.path.dirname(os.path.dirname(self.cache_path)),'cache_img_dvr')

        self.opt = opt
        self.val = val
        self.type = type

        self.scan_info = pandas.read_csv(hydra.utils.to_absolute_path(opt.data_list),dtype=str)
        self.scan_info_obj = pandas.read_csv(hydra.utils.to_absolute_path(opt.data_list_obj),dtype=str)
        self.scan_info_thrp = pandas.read_csv(hydra.utils.to_absolute_path(opt.data_list_thrp),dtype=str)

        self.n_samples = len(self.scan_info)
        self.n_samples_obj = len(self.scan_info_obj)
        self.n_samples_thrp = len(self.scan_info_thrp)

        self.names = []
        if self.opt.human:
            for i in range(len(self.scan_info)):
                self.names.append(self.scan_info.iloc[i]['id'])
        else:
            for i in range(len(self.scan_info_obj)):
                self.names.append(self.scan_info_obj.iloc[i]['id'])

        if val: 
            self.scan_info = self.scan_info[:int(self.n_samples // 25)]
            self.scan_info_obj = self.scan_info_obj[:int(self.n_samples_obj // 25)]
            # self.scan_info = self.scan_info[186:190]
            # self.scan_info_obj = self.scan_info_obj[186:190]

        self.transform = get_transform(self.opt.load_res)            

    def __getitem__(self, index):

        if self.opt.human:

            index = index//10

            scan_info = self.scan_info.iloc[index]

            batch = {}

            batch['index'] = index
            batch['type'] = int(scan_info['type'])
            batch['location'] = int(scan_info['location'])
            batch['category'] = int(scan_info['category'])


            f = np.load(os.path.join(self.dataset_path, scan_info['id'], 'occupancy.npz') )

            batch['smpl_params'] = f['smpl_params'].astype(np.float32)
            batch['smpl_betas'] =  batch['smpl_params'][76:]
            batch['smpl_thetas'] = batch['smpl_params'][4:76]

            batch['scan_name'] = str(f['scan_name'])

            batch['pts_d'] = f['pts_d']
            batch['occ_gt'] = f['occ_gt']

            if self.opt.color:
                c = np.load(os.path.join(self.dataset_path, scan_info['id'], 'color.npz'))

                batch['pts_color'] = c['sample_points']
                batch['color_gt'] = c['sample_points_color']

            if self.opt.load_surface:
                # surface_file = np.load(os.path.join(self.dataset_path, batch['scan_name'], 'surface.npz') )
                surface_file = np.load(os.path.join(self.dataset_path, scan_info['id'], 'surface.npz') )
                batch.update(surface_file)
                
            if self.opt.load_img:

                for _ in range(0, dist.get_rank()+1):
                    id_view = torch.randint(low=0,high=18,size=(1,)).item()

                batch['smpl_thetas_img'] = rectify_pose(batch['smpl_thetas'].copy(), np.array([0,2*np.pi/18.*id_view,0]))
                batch['smpl_params_img'] =  batch['smpl_params'].copy()
                batch['smpl_params_img'][4:76] = batch['smpl_thetas_img']

                # image_folder = os.path.join(self.dataset_path, batch['scan_name'], 'multi_view_%d'%(256))
                image_folder = os.path.join(self.dataset_path, scan_info['id'], 'multi_view_%d'%(256))
                batch['norm_img']= self.transform(PIL.Image.open(os.path.join(image_folder,'%05d_normal.png'%id_view)).convert('RGB'))

                if self.opt.load_cache:
                    # cache_file = np.load(os.path.join(self.cache_path, '%s.npy'%batch['scan_name']))
                    cache_file = np.load(os.path.join(self.cache_path, '%s.npy'%scan_info['id']))
                    batch['cache_pts']= cache_file[id_view,:,:,:3].reshape([-1,3])
                    batch['cache_mask']= cache_file[id_view,:,:,3].flatten().astype(bool)

        else:

            

            index = index//10
            scan_info = self.scan_info_obj.iloc[index]
            # scan_info_human = self.scan_info.iloc[index * 0]

            batch = {}

            batch['index'] = index
            # batch['type'] = int(scan_info['type'])
            # batch['location'] = int(scan_info['location'])
            batch['category'] = int(scan_info['category'])


            f = np.load(os.path.join(self.dataset_path, scan_info['id'], 'sdf.npz') )
            # f_human = np.load(os.path.join(self.dataset_path, scan_info_human['id'], 'occupancy.npz') )

            batch['smpl_params'] = f['smpl_params'].astype(np.float32)
            batch['smpl_betas'] =  batch['smpl_params'][76:]
            batch['smpl_thetas'] = batch['smpl_params'][4:76]

            # batch['smpl_params_human'] = f['smpl_params'].astype(np.float32)
            # batch['smpl_betas_human'] =  batch['smpl_params_human'][76:]
            # batch['smpl_thetas_human'] = batch['smpl_params_human'][4:76]

            batch['scan_name'] = str(f['scan_name'])

            batch['pts_d'] = f['pts_d']
            batch['sdf_gt'] = f['sdf_gt']

            if self.opt.color:
                c = np.load(os.path.join(self.dataset_path, scan_info['id'], 'color.npz'))

                batch['pts_color'] = c['sample_points']
                batch['color_gt'] = c['sample_points_color']

            if self.opt.load_surface:
                # surface_file = np.load(os.path.join(self.dataset_path, batch['scan_name'], 'surface.npz') )
                surface_file = np.load(os.path.join(self.dataset_path, scan_info['id'], 'surface.npz') )
                batch.update(surface_file)
                
            if self.opt.load_img:

                for _ in range(0, dist.get_rank()+1):
                    id_view = torch.randint(low=0,high=18,size=(1,)).item()

                batch['smpl_thetas_img'] = rectify_pose(batch['smpl_thetas'].copy(), np.array([0,2*np.pi/18.*id_view,0]))
                batch['smpl_params_img'] =  batch['smpl_params'].copy()
                batch['smpl_params_img'][4:76] = batch['smpl_thetas_img']

                # image_folder = os.path.join(self.dataset_path, batch['scan_name'], 'multi_view_%d'%(256))
                image_folder = os.path.join(self.dataset_path, scan_info['id'], 'multi_view_%d'%(256))
                batch['norm_img']= self.transform(PIL.Image.open(os.path.join(image_folder,'%05d_normal.png'%id_view)).convert('RGB'))

                if self.opt.load_cache:
                    # cache_file = np.load(os.path.join(self.cache_path, '%s.npy'%batch['scan_name']))
                    cache_file = np.load(os.path.join(self.cache_path, '%s.npy'%scan_info['id']))
                    batch['cache_pts']= cache_file[id_view,:,:,:3].reshape([-1,3])
                    batch['cache_mask']= cache_file[id_view,:,:,3].flatten().astype(bool)

    
        return batch

    def __len__(self):

        return len(self.scan_info)*10 if self.opt.human else len(self.scan_info_obj)*10


class DataProcessor():

    def __init__(self, opt):

        self.opt = opt
        self.total_points = 100000

    def process(self, batch, smpl_server, load_volume=True):

        num_batch,_,num_dim = batch['pts_d'].shape

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        batch.update(smpl_output)

        if self.opt.color:
            random_idx = torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device)
            batch['pts_color'] = torch.gather(batch['pts_color'], 1, random_idx.expand(-1, -1, num_dim))
            batch['color_gt'] = torch.gather(batch['color_gt'], 1, random_idx.expand(-1, -1, num_dim))

        if self.opt.load_img:
            
            smpl_output_img = smpl_server(batch['smpl_params_img'], absolute=False)
            smpl_output_img = { k+'_img': v for k, v in smpl_output_img.items() }
            batch.update(smpl_output_img)

        if load_volume:

            random_idx = torch.cat([torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device), # 1//8 for bbox samples
                                    torch.randint(0 ,self.total_points, [num_batch, self.opt.points_per_frame//8, 1], device=batch['pts_d'].device)+self.total_points], # 1 for surface samples
                                    1)
            batch['sdf_gt'] = torch.gather(batch['sdf_gt'], 1, random_idx)
            batch['pts_d'] = torch.gather(batch['pts_d'], 1, random_idx.expand(-1, -1, num_dim))

        if self.opt.load_surface:
            self.total_points = 100000
            random_idx = torch.randint(0, self.total_points, [num_batch, self.opt.points_per_frame, 1], device=batch['pts_d'].device)
            batch['pts_surf'] = torch.gather(batch['surface_points'], 1, random_idx.expand(-1, -1, num_dim))
            batch['norm_surf'] = torch.gather(batch['surface_normals'], 1, random_idx.expand(-1, -1, num_dim))
            
        return batch

    def process_smpl(self, batch, smpl_server):

        smpl_output = smpl_server(batch['smpl_params'], absolute=False)
        
        return smpl_output

class DataModule(pl.LightningDataModule):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        # if stage == 'fit':
        self.dataset_train = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt)
        # self.dataset_train_thrp = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt, type='thrp')
        self.dataset_val = DataSet(dataset_path=self.opt.dataset_path, opt=self.opt, val=True)
        self.meta_info = {'n_samples': self.dataset_train.n_samples,
                          'scan_info': self.dataset_train.scan_info,
                          'n_samples_obj': self.dataset_train.n_samples_obj,
                          'scan_info_obj': self.dataset_train.scan_info_obj,
                          'n_samples_gp': self.dataset_train.n_samples_thrp,
                          'scan_info_gp': self.dataset_train.scan_info_thrp,
                          'dataset_path': self.dataset_train.dataset_path}

        self.meta_info = Dict2Class(self.meta_info)

    def train_dataloader(self):

        dataloader = torch.utils.data.DataLoader(self.dataset_train,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=False)
        # dataloader_thrp = torch.utils.data.DataLoader(self.dataset_train_thrp,
        #                         batch_size=self.opt.batch_size // 2,
        #                         num_workers=self.opt.num_workers // 2, 
        #                         persistent_workers=self.opt.num_workers>0,
        #                         shuffle=True,
        #                         drop_last=True,
        #                         pin_memory=False)
        # return {"obj": dataloader, "thrp": dataloader_thrp}
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset_val,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                persistent_workers=self.opt.num_workers>0,
                                shuffle=True,
                                drop_last=False,
                                pin_memory=False)
        return dataloader




def get_transform(size):
 
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

