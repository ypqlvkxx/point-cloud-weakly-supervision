import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import ConfusionMatrix

from network.RandLANet import Network
from dataset.trainset import MarsMT
from network.lovasz import lovasz_softmax

from network.evaluation import compute_iou, compute_oa
from pytorch_lightning.loggers import TensorBoardLogger
from network.loss_helper import compute_consistency_loss, compute_Entropy_regularization_loss, compute_pseudo_loss, compute_supervised_loss

import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.spatial.distance import pdist, squareform 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class LightningEvaluator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.preds = []
        self.test_points = []
        self.config = config
        self._load_dataset_info()
        self.student = Network(self.config['model'])
        self.confusion_matrix = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))

    def forward(self, model, endpoints):
        output = model(endpoints)
        return output
    
    def test_step(self, batch, batch_idx):
        for key in batch:
            if type(batch[key]) is list:
                for i in range(self.config['model']['num_layers']):
                    batch[key][i] = batch[key][i].cuda(non_blocking=True)
            else:
                batch[key] = batch[key].cuda(non_blocking=True)
        input = batch
        label = input['labels']

        output = self(self.student, batch)
        loss = compute_supervised_loss(output, label, self.test_dataset, self.class_weights) 
        valid_logits = output['valid_logits']
        valid_labels = output['valid_labels']
        self.confusion_matrix.update(valid_logits.argmax(1), valid_labels)
        self.predict(output)

    def test_epoch_end(self, outputs):
        iou, miou, matrix = compute_iou(self.confusion_matrix.compute(), self.class_weights, ignore_zero=False)
        oa, acc_global = compute_oa(self.confusion_matrix.compute(), ignore_zero=False)
        print(matrix)
        
        self.confusion_matrix.reset()
        for class_name, class_iou in zip(self.unique_name, iou):
            print('val_iou_{}: {}'.format(class_name, class_iou * 100))
        print('val_miou:'+str(miou))
        s = 'IoU:'
        print("OA: " + str(oa))
        print("OA_global: " + str(acc_global))
        self.store()

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, 
                batch_size=self.config['test_dataloader']['batch_size'],
                shuffle=self.config['test_dataloader']['shuffle'],
                num_workers=self.config['test_dataloader']['num_workers'],
                pin_memory=self.config['test_dataloader']['pin_memory'],
                worker_init_fn=my_worker_init_fn,
                collate_fn=self.test_dataset.collate_fn)
    
    def predict(self, end_points):
        # Store logits into list
        B = end_points['logits'].size(0)
        for j in range(B):
            probs = end_points['logits'][j].cpu().numpy()
            point = end_points['xyz'][0][j].cpu().numpy()
            pred = np.argmax(probs, 0).astype(np.uint32)


            self.preds.append(pred)
            self.test_points.append(point)
        

    def store(self):
        # initialize result directory
        root_dir = os.path.join(self.config['dataset']['prediction_root_dir'])
        os.makedirs(root_dir, exist_ok=True)
        self.preds = np.stack(self.preds)
        self.test_points = np.stack(self.test_points)
        N = len(self.preds)
        for j in range(N):    
            pred = self.preds[j]
            pred += 1
            # 0 - 19    
            name = self.test_dataset.data_list[j][1] + '.txt'
            save_point=self.test_points[j]
            output_path = os.path.join(root_dir, name)
            pred_point=np.concatenate((save_point,pred[:,np.newaxis]),axis=1)
            np.savetxt(output_path, pred_point)
    
    def setup(self, stage):
        self.test_dataset = MarsMT(split='test', config=self.config['dataset'])
        # self.class_weights = [1 for _ in range(self.nclasses)]
        # self.class_weights = np.expand_dims(self.class_weights, axis=0)
        # self.class_weights = torch.from_numpy(np.array(self.class_weights)).float().cuda()
        self.class_weights = torch.from_numpy(self.test_dataset.get_class_weight()).float().cuda()

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = self.config['dataset']['numclass']
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'])
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_miou:.2f}',
                                                  monitor='val_miou', mode='max', save_top_k=3)
        return [checkpoint]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/Mars_training.yaml')
    parser.add_argument('--dataset_config_path', default='config/datasetconfig.yaml')
    parser.add_argument('--ckpt_path', default='/home/checkpoint.ckpt')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    trainer = Trainer(**config['trainer'])
    model = LightningEvaluator.load_from_checkpoint(args.ckpt_path, config=config, strict=False)
    trainer.test(model)