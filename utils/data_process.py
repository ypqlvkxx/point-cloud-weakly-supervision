from os.path import join
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import utils.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors #

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class DataProcessing:

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)
    
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def get_file_list(dataset_path, seq_list):
        data_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)#标签位置
            pc_path = join(seq_path, 'velodyne')#点位置
            new_data = [(seq_id, f[:-4]) for f in np.sort(os.listdir(pc_path))]
            data_list.extend(new_data)

        return data_list
    
    @staticmethod
    def get_Mars_file_list(dataset_path, list):
        data_list = []
        for id in list:
            #标签位置
            pc_path = join(dataset_path,id)#点和标签位置
            new_data = [(id, f[:-4]) for f in np.sort(os.listdir(pc_path))]
            data_list.extend(new_data)

        return data_list

    def get_active_list(list_root):
        train_list = []
        pool_list = []
        with open(join(list_root, 'label_data.json')) as f:
            train_list = json.load(f)
        with open(join(list_root, 'ulabel_data.json')) as f:
            pool_list = json.load(f)
        pool_list += train_list
        train_list = []
        return train_list, pool_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(data_root, paths, num_of_class):
        # pre-calculate the number of points in each category
        num_per_class = [0 for _ in range(num_of_class)]
        for file_path in tqdm(paths, total=len(paths)):
            label_path = join(data_root, file_path[0],file_path[1]+ '.txt')
            label = np.loadtxt(label_path)[:,4].astype(int).reshape(-1)
            inds, counts = np.unique(label, return_counts=True)
            for i, c in zip(inds, counts):
                if i == 0:      # 0 : unlabeled
                    continue
                else:
                    num_per_class[i-1] += c
        #scribble
        # num_per_class = np.array([13825,14842,16055,0,71503,
        #                               41291,0,0,0,12126,5657,99282,278,321,
        #                               91608,37131,0,18213])
        
        #Full
        # num_per_class = np.array([193110,1038776,114860,0,717128,
        #                              484677,0,0,0,151394,66468,1354603,3491,1462,
        #                              1110107,776860,0,217901])
        '''
        if dataset_name == 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name == 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name == 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        if dataset_name == 'Mars':
            num_per_class = np.array([4533411,3371869], dtype=np.int32)
        '''
        # num_per_class = np.array([4533411,3371869], dtype=np.int32)
        num_per_class = np.array(num_per_class)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        #对权重进行归一化操作
        indices_to_zero = np.where(ce_label_weight == 50)[0]  
        #indices_to_one = np.where(ce_label_weight != 50)[0]  
        # 将这些索引对应的值替换为0  
        ce_label_weight[indices_to_zero] = 0 
        num = len(ce_label_weight) - len(indices_to_zero)
        #ce_label_weight[indices_to_one] = 1.0 
        ce_label_weight = ce_label_weight * num / float(sum(ce_label_weight))
        # ce_label_weight = np.array([0.9922, 0.9489, 0.9019, 0, 0.2764, 0.4443, 0, 0,
        #                               0, 1.0742, 1.5611, 0.2051, 2.5337, 2.5213, 0.2208,
        #                               0.48483, 0, 0.8289])
        return np.expand_dims(ce_label_weight, axis=0)
