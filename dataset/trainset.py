import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_process import DataProcessing as DP
from os.path import join
import numpy as np
import torch


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_color(color):
    centroid = np.mean(color)
    color = color - centroid
    m = np.max(np.sqrt(np.sum(color ** 2)))
    color = color / m
    return color

class MarsMT(Dataset):
    def __init__(self, split, data_list=None, config=None):
        self.config = config
        self.name = 'Mars_Terrain'
        self.dataset_path = self.config['root_dir']
        self.label_to_names = {0: 'unlabeled',
                               1: 'crater',
                               2: 'crater_eject',
                               3: 'channel',
                               4: 'delta',
                               5: 'mesa',
                               6: 'polygons',
                               7: 'wrinkle_ridges',
                               8: 'lava_tube',
                               9: 'lava_flow',
                               10: 'tranverse_sand_ridges',
                               11: 'curve_dunes',
                               12: 'yardangs',
                               13: 'gullies',
                               14: 'dark_slope_streak',
                               15: 'mound',
                               16: 'ridge',
                               17: 'cliff',
                               18: 'smooth_terrain'
                               }
        self.split = split


        self.num_classes = self.config['numclass']
        self.ignored_labels = np.sort([0])
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.spilt = split
        self.mode = split
        if data_list is None:
            if split == 'train':
                list = ['train']
            elif split == 'valid':
                list = ['val']
            elif split == 'test':
                list = ['test']
            self.data_list = DP.get_Mars_file_list(self.dataset_path, list)
        else:
            self.data_list = data_list
        self.data_list = sorted(self.data_list)

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, item):
        return {
            'full': self.spatially_regular_gen(item, self.data_list, self.config['aug']['full'])
        }


    def spatially_regular_gen(self, item, data_list, aug_methods):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, color, labels = self.get_data(pc_path)
        # crop a small point cloud
        #
        selected_pc, selected_colors, selected_labels, selected_idx = self.crop_pc(pc, color, labels)
        if self.split == 'train':
            selected_pc = self.augment(selected_pc, aug_methods)

        return selected_pc, selected_colors, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        #seq_id = file_path[0]
        frame_id = file_path[1]
        points_path=join(self.dataset_path, file_path[0],frame_id + '.txt')
        label_points = np.loadtxt(points_path)
        # load labels
        #label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        points=label_points[:,0:3]
        color = label_points[:,3]
        labels=label_points[:,4]
        return points, color, labels

    @staticmethod
    def crop_pc(points, colors, labels):
        # crop a fixed size point cloud for training
        #center_point = points[pick_idx, :].reshape(1, -1)
        #select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        select_idx = np.indices(labels.shape)[0,:]
        select_points = points[select_idx]
        select_points = pc_normalize(select_points)
        select_colors = colors[select_idx]
        select_colors = pc_normalize_color(select_colors)
        select_labels = labels[select_idx] 
        return select_points, select_colors, select_labels, select_idx
    
    @staticmethod
    def augment(xyz, methods):
        if 'rotate' in methods:
            angle = np.deg2rad(np.random.random()*90) - np.pi/4
            c, s = np.cos(angle), np.sin(angle)
            R = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot(xyz[:,:2], R)

        if 'flip' in methods:
            direction = np.random.choice(4,1)
            if direction == 1:
                xyz[:,0] = -xyz[:,0]
            elif direction == 2:
                xyz[:,1] = -xyz[:,1]
            elif direction == 3:
                xyz[:,:2] = -xyz[:,:2]

        if 'scale' in methods:
            s = np.random.uniform(0.95, 1.05)
            xyz[:,:2] = s * xyz[:,:2]

        if 'noise' in methods:
            noise = np.array([np.random.normal(0, 0.0001, 1),
                              np.random.normal(0, 0.0001, 1),
                              np.random.normal(0, 0.0001, 1)]).T
            xyz[:,:3] += noise
        return xyz

    def tf_map(self, batch_pc, batch_color, batch_label, batch_pc_idx, batch_cloud_idx):
        features = np.concatenate([batch_pc, batch_color[:,:,np.newaxis]], axis=-1)
        #features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.config['num_layers']):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, self.config['k_n'])
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.config['sub_sampling_ratio'][i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // self.config['sub_sampling_ratio'][i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):
        input_fea = {}
        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind = [], [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i]['full'][0])
            selected_colors.append(batch[i]['full'][1])
            selected_labels.append(batch[i]['full'][2])
            selected_idx.append(batch[i]['full'][3])
            cloud_ind.append(batch[i]['full'][4])

        selected_pc = np.stack(selected_pc)
        selected_colors = np.stack(selected_colors)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind)

        num_layers = self.config['num_layers']
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['idx'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()
        
        input_fea=inputs

        return input_fea

    