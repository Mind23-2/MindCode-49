import os
import argparse
from mindspore import context
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
import mindspore.nn as nn
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from easydict import EasyDict

from src.dataset import create_dataset_Cifar10 as Cifar10
from src.dataset import create_dataset_ImageNet as ImageNet
from src.lr_generator import get_lr, power_lr
from src.glore_resnet50 import glore_resnet50

parser = argparse.ArgumentParser(description='Image classification with glore_res50 on cifar10')
parser.add_argument('--dataset_path', type=str, default='/opt_data/xidian_wks/yqd/lenet/dataset/train/',
                    help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['GPU', 'Ascend'])
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--pre_trained', type=str, default=None)

args = parser.parse_args()

config = EasyDict({
    "class_num": 10,
    "batch_size": 8,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 90,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr_decay_mode": "poly",
    "lr_init": 0.01,
    "lr_end": 0.00001,
    "lr_max": 0.1
})

if __name__ == '__main__':
    target = args.device_target
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        save_graphs=False,
                        device_id=device_id)
    # create dataset
    # train_dataset = Cifar10(dataset_path=args.dataset_path, do_train=True, repeat_num=1,
    #                         batch_size=config.batch_size, target=target)

    train_dataset = ImageNet(dataset_path=args.dataset_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target)
    step_size = train_dataset.get_dataset_size()

    # init lr
    lr = get_lr(lr_init=config.lr_init,
                lr_end=config.lr_end,
                lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size,
                steps_per_epoch=train_dataset.get_dataset_size(),
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)
    # define net
    net = glore_resnet50(num_classes=config.class_num, use_glore=True)

    # define opt
    # net_opt = nn.SGD(params=net.trainable_params(), learning_rate=lr, weight_decay=0.0002)
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
    net_opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=net_opt, loss_scale_manager=loss_scale, metrics={'acc'})

    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps= \
                                         config.save_checkpoint_epochs * \
                                         train_dataset.get_dataset_size(),
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", config=config_ck)
        cb += [ckpt_cb]
    for m in net.cells_and_names():
        print(m[0])
    print("\n\n========================")
    print("Dataset path: {}".format(args.dataset_path))
    print("Total epoch: {}".format(config.epoch_size))
    print("Batch size: {}".format(config.batch_size))
    print("Class num: {}".format(config.class_num))
    print("=======Training begin========")
    model.train(config.epoch_size - config.pretrain_epoch_size, train_dataset,
                callbacks=cb, dataset_sink_mode=True)
