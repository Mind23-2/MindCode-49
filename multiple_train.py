from mindspore import context
import os
import random
import argparse
import ast
import numpy as np
from mindspore import Tensor
from mindspore import dataset as de
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from easydict import EasyDict
import moxing as mox

from src.lr_generator import get_lr
from src.glore_resnet50 import glore_resnet50
from src.dataset import create_dataset_ImageNet as ImageNet
from src.dataset import create_dataset_Cifar10 as Cifar10


parser = argparse.ArgumentParser(description='Image classification with glore_resnet50')
parser.add_argument('--use_glore', type=bool, default=True, help='Enable GloreUnit')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=True, help='Run distribute')
parser.add_argument('--device_num', type=int, default=8, help='Device num.')
parser.add_argument('--data_url', type=str, default='/opt_data/xidian_wks/imagenet_original/train/',
                    help='Dataset path')
parser.add_argument('--train_url', type=str)
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=bool, default=False)
parser.add_argument('--pre_ckpt_path', type=str,
                    default='')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
args_opt = parser.parse_args()

config = EasyDict({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.92,
    "weight_decay": 0.0001,
    "epoch_size": 180,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "steps",
    "lr_init": 0.1,
    "lr_end": 0,
    "lr_max": 0.4
})


class SoftmaxCrossEntropyExpand(nn.Cell):  # pylint: disable=missing-docstring
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()
        self.eps = Tensor(1e-24, mstype.float32)

    def construct(self, logit, label):  # pylint: disable=missing-docstring
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result + self.eps)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss


if __name__ == '__main__':
    target = args_opt.device_target
    
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              auto_parallel_search_mode="recursive_programming")
            init()
    # create dataset
    
    # download dataset from obs to cache
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='/cache/data_mmq')
    dataset_path = '/cache/data_mmq'
    
    dataset = ImageNet(dataset_path=dataset_path,
                       do_train=True,
                       use_randaugment=True,
                       repeat_num=1,
                       batch_size=config.batch_size,
                       target=target)

#     dataset = Cifar10(dataset_path=dataset_path, do_train=True, repeat_num=1,
#                       batch_size=config.batch_size, target=target)

    step_size = dataset.get_dataset_size()

    # define net

    net = glore_resnet50(num_classes=config.class_num, use_glore=args_opt.use_glore)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_ckpt_path)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)

#                 cell.weight.default_input = weight_init.initializer(weight_init.HeNormal(mode='fan_out', ),
#                                                                     cell.weight.shape,
#                                                                     cell.weight.dtype)

            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.shape,
                                                                    cell.weight.dtype)

    # init lr
    # lr = power_lr(0.4, config.epoch_size, step_size)
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    #
    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    net_opt = nn.SGD(group_params, learning_rate=lr, momentum=config.momentum,
                     weight_decay=config.weight_decay, loss_scale=config.loss_scale,
                     nesterov=True)
    # net_opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    # define loss, model
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="glore_resnet50", directory='/cache/train_output/device_' +
                                  os.getenv('DEVICE_ID') + '/', config=config_ck)
        cb += [ckpt_cb]
   
    # train model
    print("===========================================")
    print("Total epoch: {}".format(config.epoch_size))
    print("Class num: {}".format(config.class_num))
    print("Backbone resnet50")
    print("Enable glore: {}".format(args_opt.use_glore))
    print("=======Multiple Training Begin========")
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
                callbacks=cb, dataset_sink_mode=True)
    
    # copy train result from cache to obs
    mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args_opt.train_url)
