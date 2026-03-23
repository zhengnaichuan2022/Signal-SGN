#!/usr/bin/env python
"""
Signal-SGN Training Script
"""
from __future__ import print_function
import torch.nn.utils as utils
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
from spikingjelly.activation_based import neuron, layer, functional
import numpy as np
import glob
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    parser = argparse.ArgumentParser(description='Signal-SGN: Skeleton-based Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir', help='work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='config/default.yaml', help='path to configuration file')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='store classification score')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, help='print interval (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='save interval (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=0, help='start epoch to save model')
    parser.add_argument('--eval-interval', type=int, default=5, help='evaluate interval (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='Top K accuracy')
    parser.add_argument('--feeder', default='feeders.feeder_ntus.Feeder', help='data loader')
    parser.add_argument('--num-worker', type=int, default=16, help='data loader workers')
    parser.add_argument('--train-feeder-args', default=dict(), help='training data loader args')
    parser.add_argument('--test-feeder-args', default=dict(), help='test data loader args')
    parser.add_argument('--model', default=None, help='model')
    parser.add_argument('--model-args', default=dict(), help='model args')
    parser.add_argument('--weights', default="", help='weights for initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='ignore weights')
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='lr decay epochs')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='GPU indexes')
    parser.add_argument('--optimizer', default='SGD', help='optimizer type')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='total epochs')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='lr decay rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--T', default=10, type=int)
    return parser


class Processor:
    """Processor for Skeleton-based Action Recognition with Signal-SGN"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args.get('debug', False):
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir:', arg.model_saved_name, 'already exist')
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()

        if arg.phase != 'model_size':
            self.load_optimizer()
            self.load_data()
        self.lr = arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)
        if isinstance(arg.device, list) and len(arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        if os.path.exists(self.arg.work_dir):
            shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1]) if '.pt' in self.arg.weights else 0
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            model_state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()
                                   if k.split('module.')[-1] in model_state_dict])

            for w in self.arg.ignore_weights:
                keys = list(weights.keys())
                for key in keys:
                    if w in key and key in weights:
                        weights.pop(key, None)

            try:
                self.model.load_state_dict(weights, strict=False)
            except RuntimeError as e:
                self.print_log('Error loading weights: {}'.format(e))

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=self.arg.base_lr * 0.1)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if epoch < self.arg.warm_up_epoch:
            lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
        else:
            lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def _get_model_output(self, data):
        """Model returns (logits, beta) or logits only."""
        output = self.model(data)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                timer['dataloader'] += self.split_time()

            output = self._get_model_output(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            self.optimizer.step()
            functional.reset_net(self.model)

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            _, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())

            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        self.print_log('\tMean training loss: {:.4f}. Mean training acc: {:.2f}%.'.format(
            np.mean(loss_value), np.mean(acc_value) * 100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if epoch > 20:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=True, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file:
            f_w = open(wrong_file, 'w')
        if result_file:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            acc_value = []
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    label = label.long().cuda(self.output_device)
                    data = data.float().cuda(self.output_device)
                    output = self._get_model_output(data)
                    loss = F.cross_entropy(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    functional.reset_net(self.model)
                if wrong_file or result_file:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy:', accuracy, 'model:', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                self.eval(epoch, save_score=True, loader_name=['test'])

            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))
            if weights_path:
                weights_path = weights_path[0]
                weights = torch.load(weights_path)
                if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
                self.model.load_state_dict(weights)
                wf = weights_path.replace('.pt', '_wrong.txt')
                rf = weights_path.replace('.pt', '_right.txt')
                self.arg.print_log = False
                self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
                self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model total params: {num_params}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.print_log('Model: {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
            else:
                parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
