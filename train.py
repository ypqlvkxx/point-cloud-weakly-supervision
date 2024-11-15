import os
from turtle import onrelease
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
from network.evaluation import compute_iou
from pytorch_lightning.loggers import TensorBoardLogger
from network.loss_helper import (compute_consistency_loss, compute_Entropy_regularization_loss
                                        , compute_pseudo_loss, compute_supervised_loss, compute_contra_memobank_loss)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.net = Network(self.config['model'])
        self.confusion_matrix = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))

        self.prediction = []
        self.ensemble_prediction ={}
        self.ensemble_representation = {}

        self.save_hyperparameters('config')

    def forward(self, model, endpoints):
        output = model(endpoints)
        return output

    def training_step(self, batch, batch_idx):
        def get_ingore_mask(label):
            ignored_bool = (label == 0)
            #
            for ign_label in self.train_dataset.ignored_labels:
                ignored_bool = ignored_bool | (label == ign_label)
            return ignored_bool

        lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log('learning_rate', lr, on_epoch=True, prog_bar=True)

        if self.current_epoch == 0:
            batch = self.batch_to_cuda(batch)
            input = batch
            label = input['labels']
            output = self(self.net, batch)
            #compute loss
            sup_loss = (
                compute_supervised_loss(
                    output, 
                    label, 
                    self.train_dataset, 
                    self.class_weights
                    )
            )
            loss = sup_loss
            self.EMA(batch, 0.9)

        
        elif self.current_epoch > 0:
            batch = self.batch_to_cuda(batch)
            #forward 
            input = batch
            label = input['labels'] 
            #inference
            output = self(self.net, batch)
            #compute loss
            with torch.no_grad():
                ignore_mask = get_ingore_mask(label) 
            sup_loss = (
                compute_supervised_loss(
                    output, 
                    label, 
                    self.train_dataset, 
                    self.class_weights
                    )
            )
            re_loss = (
                compute_Entropy_regularization_loss(
                    output, 
                    mask = ignore_mask
                    )
            )
            con_loss = (
                compute_consistency_loss(
                    output, 
                    self.ensemble_prediction, 
                    mask = ignore_mask
                    )
            )
            self.EMA(batch, 0.9)
            if self.current_epoch <= 150 and self.current_epoch > 0:
                lamida = np.math.exp(-5*np.math.pow((1.5-(self.current_epoch/150)),2))
                loss = sup_loss + lamida * re_loss + lamida * con_loss
            elif self.current_epoch > 150:
                lamida = 1
                loss = sup_loss + lamida * re_loss + lamida * con_loss 

            self.log('train_loss_s', sup_loss, on_epoch=True, prog_bar=True)
            self.log('train_loss_re', lamida * re_loss, on_epoch=True, prog_bar=True)
            self.log('train_loss_c', lamida * con_loss, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        return loss


    def logits_to_probs(self, batch):
        prediction = []
        B = batch['logits'].size(0)
        for j in range(B):
            probs = batch['logits'][j]
            # 使用softmax得到概率分布  
            probs = torch.nn.functional.softmax(probs, dim=0)  
            prediction.append(probs.detach().cpu().numpy())
        
        return prediction
    
    def EMA(self, batch, alpha = 0.9):
        preds = self.logits_to_probs(batch)
        for id_, prob, rep in zip(batch['idx'].cpu().numpy(), preds):  
            if id_[0] in self.ensemble_prediction:  
                self.ensemble_prediction[id_[0]] = self.ensemble_prediction[id_[0]] * alpha + prob * (1 - alpha)  
            else:
                self.ensemble_prediction[id_[0]] = prob  

    def batch_to_cuda(self, batch):
        for key in batch:
                if type(batch[key]) is list:
                    for i in range(self.config['model']['num_layers']):
                        batch[key][i] = batch[key][i].cuda(non_blocking=True)
                else:
                    batch[key] = batch[key].cuda(non_blocking=True)
        return batch
    
    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), **self.config['optimizer'])
        milestones = [180, 210, 250, 300, 420]
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict

    def setup(self, stage):
        self.train_dataset = MarsMT(split='train', config=self.config['dataset'])
        self.test_dataset = MarsMT(split='test', config=self.config['dataset'])
        self.class_weights = torch.from_numpy(self.train_dataset.get_class_weight()).float().cuda()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, 
                batch_size=self.config['train_dataloader']['batch_size'],
                shuffle=self.config['train_dataloader']['shuffle'],
                num_workers=self.config['train_dataloader']['num_workers'],
                pin_memory=self.config['train_dataloader']['pin_memory'],
                worker_init_fn=my_worker_init_fn,
                collate_fn=self.train_dataset.collate_fn)

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = self.config['dataset']['numclass']
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'])
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}',every_n_train_steps=10, 
                                                  save_top_k=-1)
        return [checkpoint]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/Mars_training.yaml')
    parser.add_argument('--dataset_config_path', default='config/datasetconfig.yaml')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))

    #Create TensorBoardLogger
    tensorboard_logger = TensorBoardLogger(
                                            save_dir=config['trainer']['default_root_dir'],
                                            name=config['logger']['name']) 

    
    model = LightningTrainer(config)

    CHECKPOINT_PATH = config['checkpoint']['path']
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        Trainer(logger=tensorboard_logger,
                callbacks=model.get_model_callback(),
                resume_from_checkpoint = CHECKPOINT_PATH,
                **config['trainer']).fit(model)
    else:
        Trainer(logger=tensorboard_logger,
                callbacks=model.get_model_callback(),
                **config['trainer']).fit(model)

    