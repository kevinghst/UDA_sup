# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
# 
# Except load and save function, the whole codes of file has been modified and added by
# SanghunYun, Korea University for UDA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from copy import deepcopy
from typing import NamedTuple
from tqdm import tqdm
import time
import shutil
from logger import *

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from utils import checkpoint
# from utils.logger import Logger
from tensorboardX import SummaryWriter
from utils.utils import output_logging, bin_accuracy, multi_accuracy, AverageMeterSet
import pdb


class Trainer(object):
    """Training Helper class"""
    def __init__(self, cfg, model, data_iter, optimizer, device, ema_model, ema_optimizer):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ema_model = ema_model
        self.ema_optimizer = ema_optimizer

        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file):

        if self.cfg.uda_mode or self.cfg.mixmatch_mode:
            ssl_mode = True
        else:
            ssl_mode = False
        """ train uda"""

        # tensorboardX logging
        if self.cfg.results_dir:
            dir = os.path.join('results', self.cfg.results_dir)
            if os.path.exists(dir) and os.path.isdir(dir):
                shutil.rmtree(dir)

            writer = SummaryWriter(log_dir=dir)

            #logger_path = dir + 'log.txt'
            #logger = Logger(logger_path, title='uda')
            #if self.cfg.no_unsup_loss:
            #    logger.set_names(['Train Loss', 'Valid Acc', 'Valid Loss', 'LR'])
            #else:
            #    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss W U', 'Valid Acc', 'Valid Loss', 'LR'])
            
        meters = AverageMeterSet()

        self.model.train()
        self.load(model_file, pretrain_file)    # between model_file and pretrain_file, only one model will be loaded
        model = self.model.to(self.device)
        ema_model = self.ema_model.to(self.device) if self.ema_model else None

        if self.cfg.data_parallel:                       # Parallel GPU mode
            model = nn.DataParallel(model)
            ema_model = nn.DataParallel(ema_model) if ema_model else None

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0, 0., 0.]   # acc, step, val_loss, train_loss
        no_improvement = 0

        sup_batch_size = None
        unsup_batch_size = None

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps, disable=self.cfg.hide_tqdm) if ssl_mode \
              else tqdm(self.sup_iter, total=self.cfg.total_steps, disable=self.cfg.hide_tqdm)

        for i, batch in enumerate(iter_bar):
            # Device assignment
            if ssl_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]

                unsup_batch_size = unsup_batch_size or unsup_batch[0].shape[0]

                if unsup_batch[0].shape[0] != unsup_batch_size:
                    continue
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None

            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss, weighted_unsup_loss = get_loss(model, sup_batch, unsup_batch, global_step)

            if self.cfg.no_sup_loss:
                final_loss = unsup_loss
            elif self.cfg.no_unsup_loss:
                final_loss = sup_loss

            meters.update('train_loss', final_loss.item())
            meters.update('sup_loss', sup_loss.item())
            meters.update('unsup_loss', unsup_loss.item())
            meters.update('w_unsup_loss', weighted_unsup_loss.item())
            meters.update('lr', self.optimizer.get_lr()[0])

            final_loss.backward()
            self.optimizer.step()

            if self.ema_optimizer:
                self.ema_optimizer.step()

            # print loss
            global_step += 1
            loss_sum += final_loss.item()
            if not self.cfg.hide_tqdm:
                if ssl_mode:
                    iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f'\
                            % (final_loss.item(), unsup_loss.item(), sup_loss.item()))
                else:
                    iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)

            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > self.cfg.check_after:
                if self.cfg.mixmatch_mode:
                    results = self.eval(get_acc, None, ema_model)
                else:
                    total_accuracy, avg_val_loss = self.validate()

                # logging
                writer.add_scalars('data/eval_acc', {'eval_acc' : total_accuracy}, global_step)
                writer.add_scalars('data/eval_loss', {'eval_loss': avg_val_loss}, global_step)

                if self.cfg.no_unsup_loss:
                    writer.add_scalars('data/train_loss', {'train_loss': meters['train_loss'].avg}, global_step)
                    writer.add_scalars('data/lr', {'lr': meters['lr'].avg}, global_step)
                else:
                    writer.add_scalars('data/train_loss', {'train_loss': meters['train_loss'].avg}, global_step)
                    writer.add_scalars('data/sup_loss', {'sup_loss': meters['sup_loss'].avg}, global_step)
                    writer.add_scalars('data/unsup_loss', {'unsup_loss': meters['unsup_loss'].avg}, global_step)
                    writer.add_scalars('data/w_unsup_loss', {'w_unsup_loss': meters['w_unsup_loss'].avg}, global_step)
                    writer.add_scalars('data/lr', {'lr': meters['lr'].avg}, global_step)

                meters.reset()

                if max_acc[0] < total_accuracy:
                    self.save(global_step)
                    max_acc = total_accuracy, global_step, avg_val_loss, final_loss.item()
                    no_improvement = 0
                else:
                    no_improvement += 1

                print("  Top 1 Accuracy: {0:.4f}".format(total_accuracy))
                print("  Validation Loss: {0:.4f}".format(avg_val_loss))
                print("  Train Loss: {0:.4f}".format(final_loss.item()))
                if ssl_mode:
                    print("  Sup Loss: {0:.4f}".format(sup_loss.item()))
                    print("  Unsup Loss: {0:.4f}".format(unsup_loss.item()))
                print("  Learning rate: {0:.7f}".format(self.optimizer.get_lr()[0]))

                print(
                    'Max Accuracy : %5.3f Best Val Loss : %5.3f Best Train Loss : %5.4f Max global_steps : %d Cur global_steps : %d' 
                    %(max_acc[0], max_acc[2], max_acc[3], max_acc[1], global_step), end='\n\n'
                )
                
                if no_improvement == self.cfg.early_stopping:
                    print("Early stopped")
                    break


            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum/(i+1)))
                if get_acc:
                    if self.cfg.mixmatch_mode:
                        results = self.eval(get_acc, None, ema_model)
                    else:
                        total_accuracy, avg_val_loss = self.validate()
                    if max_acc[0] < total_accuracy:
                        max_acc = total_accuracy, global_step, avg_val_loss, final_loss.item()             
                    print("  Top 1 Accuracy: {0:.4f}".format(total_accuracy))
                    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                    print("  Train Loss: {0:.2f}".format(final_loss.item()))
                    print('Max Accuracy : %5.3f Best Val Loss :  %5.3f Best Train Loss :  %5.3f Max global_steps : %d Cur global_steps : %d' %(max_acc[0], max_acc[2], max_acc[3], max_acc[1], global_step), end='\n\n')
                self.save(global_step)
                return
        writer.close()
        return global_step


    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        model.eval()

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results
            

    def validate(self):
        t0 = time.time()

        print("Running validation")

        model = self.model
        device = self.device
        val_loader = self.eval_iter
        cfg = self.cfg

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        total_prec1 = 0
        total_prec5 = 0

        # Evaluate data for one epoch
        for batch in val_loader:
            batch = [t.to(device) for t in batch]
            b_input_ids, b_input_mask, b_segment_ids, b_labels = batch
            batch_size = b_input_ids.size(0)

            with torch.no_grad():        
                logits = model(
                    b_input_ids,
                    b_segment_ids,
                    b_input_mask
                )
                    
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, b_labels)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.

            if cfg.num_labels == 2:
                logits = logits.detach().cpu().numpy()
                b_labels = b_labels.to('cpu').numpy()
                total_prec1 += bin_accuracy(logits, b_labels)
            else:
                prec1, prec5 = multi_accuracy(logits, b_labels, topk=(1,5))
                total_prec1 += prec1
                total_prec5 += prec5

        avg_prec1 = total_prec1 / len(val_loader)
        avg_prec5 = total_prec5 / len(val_loader)

        avg_val_loss = total_eval_loss / len(val_loader)
        

        return avg_prec1, avg_val_loss


    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                )   # load only transformer parts
    
    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join('results', self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join('results', self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                        os.path.join('results', self.cfg.results_dir, 'save', 'model_steps_'+str(i)+'.pt'))

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x