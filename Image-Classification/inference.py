# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT training and evaluating script
This script is modified from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""

import time
import os
import logging
from contextlib import suppress
import vitae
import vitaev2

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import resolve_data_config
from timm.models import load_checkpoint, create_model
from timm.utils import *
from Utils.samplers import create_loader
from Utils import build_dataset

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

import utils

LABELS = []

with open("/workspace/Image-Classification/labels.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        LABELS.append(line.split("\n")[0].replace("\t",""))

try:
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def load_model(args):
    setup_default_logging()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    torch.manual_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp,
        img_size=args.img_size)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))
        _logger.info('Model %s created, trainable param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters() if m.requires_grad == True])))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            _logger.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    model_ema_decay = args.model_ema_decay ** (args.batch_size * utils.get_world_size() // 512.0)
    args.model_ema_decay = model_ema_decay
    if args.distributed:
        if args.sync_bn:
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    load_checkpoint(model, args.eval_checkpoint, args.model_ema, strict=True)
    model.eval()

    return model

def run(model, args, eval_dir, amp_autocast=suppress):

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    dataset_eval = build_dataset(eval_dir, idxFilesRoot='', ZIP_MODE=False)

    loader = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            output = output.softmax(-1)

            output, indices = output.topk(1)
            np_indices = indices.cpu().numpy()
            return {"label": LABELS[np_indices[0][0]]}

if __name__ == '__main__':

    from argparse import Namespace
    imgs_path = 'testset/val'
    args = Namespace(amp=False,
                    apex_amp=False,
                    batch_size=64,
                    bn_eps=None,
                    bn_momentum=None,
                    bn_tf=False,
                    channels_last=False,
                    clip_grad=None,
                    crop_pct=None,
                    decay_rate=0.1,
                    device='cuda:0',
                    dist_bn='',
                    distributed=False,
                    drop=0.0,
                    drop_block=None,
                    drop_connect=None,
                    drop_path=0.1,
                    eval_checkpoint='/workspace/Image-Classification/ViTAE-T.pth.tar',
                    eval_metric='top1',
                    gp=None,
                    img_size=224,
                    interpolation='',
                    local_rank=0,
                    log_interval=50,
                    mean=None,
                    model='ViTAE_basic_Tiny',
                    model_ema=True,
                    model_ema_decay=1.0,
                    model_ema_force_cpu=False,
                    momentum=0.9,
                    native_amp=False,
                    no_prefetcher=False,
                    num_classes=1000,
                    num_gpu=1,
                    opt='adamw',
                    opt_betas=None,
                    opt_eps=None,
                    output='',
                    pin_mem=False,
                    prefetcher=True,
                    pretrained=False,
                    rank=0,
                    real_labels='./images/real.json',
                    recovery_interval=0,
                    results_file='',
                    save_images=False,
                    seed=42,
                    smoothing=0.1,
                    split_bn=False,
                    std=None,
                    sync_bn=False,
                    train_interpolation='random',
                    use_multi_epochs_loader=False,
                    validation_batch_size_multiplier=1,
                    weight_decay=0.05,
                    workers=8,
                    world_size=1)

    start = time.time()


    model = load_model(args)

    time_taken = round(time.time() - start, 2)
    print(f"Time taken {time_taken}")
    start = time.time()

    label = run(model, args, imgs_path)

    print(label)
    time_taken = round(time.time() - start, 2)
    print(f"Time taken {time_taken}")
