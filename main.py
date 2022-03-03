import os
import argparse
import logging as logger
from resnet_def import create_net
from ffc_ddp import FFC
import torch
from torch.utils.data import DataLoader
from util import *
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from optim.optimizer import get_optim_scheduler
import torch.distributed as dist
import random
from torch.nn.parallel import DistributedDataParallel as ddp
import time
logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')



def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(id_loader, instance_loader, ffc_net, optimizer,
                    cur_epoch, conf, saved_dir, real_iter, scaler, lr_policy, lr_scheduler, warmup_epochs, max_epochs):
    """Tain one epoch by traditional training.
    """
    id_iter = iter(id_loader)
    random.seed(cur_epoch)
    avg_data_load_time = 0
    my_rank = dist.get_rank()
    db_size = len(instance_loader)
    start_time = time.time()
    for batch_idx, (ins_images, instance_label, _) in enumerate(instance_loader):
        # Note that label lies at cpu not gpu !!!
        # start_time = time.time()
        if lr_policy != 'ReduceLROnPlateau':
            lr_scheduler.update(None, batch_idx * 1.0 / db_size)
        instance_images = ins_images.cuda(conf.local_rank, non_blocking=True)
        try:
            images1, images2, id_indexes = next(id_iter)
        except:
            id_iter = iter(id_loader)
            images1, images2, id_indexes = next(id_iter)

        images1_gpu = images1.cuda(conf.local_rank, non_blocking=True)
        images2_gpu = images2.cuda(conf.local_rank, non_blocking=True)

        instance_images1, instance_images2 = torch.chunk(instance_images, 2)
        instance_label1, instance_label2 = torch.chunk(instance_label, 2)


        optimizer.zero_grad()
        x = torch.cat([images1_gpu, instance_images1])
        y = torch.cat([images2_gpu, instance_images2])
        x_label = torch.cat([id_indexes, instance_label1])
        y_label = torch.cat([id_indexes, instance_label2])

        with torch.cuda.amp.autocast():
            loss = ffc_net(x, y, x_label, y_label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        real_iter += 1
        if real_iter % 1000 == 0:
            loss_val = loss.item()
            lr = lr_scheduler.get_lr()[0]
            duration = time.time() - start_time
            left_time = (max_epochs * db_size - real_iter) * (duration / 1000) / 3600
            logger.info('Iter %d Loss %.4f Epoch %d/%d Iter %d/%d Left %.2f hours' % (real_iter, loss_val, cur_epoch, max_epochs, batch_idx + 1, db_size, left_time))
            if lr_policy == 'ReduceLROnPlateau':
                lr_scheduler.step(loss_val)
            start_time = time.time()

        if real_iter % 2000 == 0 and cur_epoch >= 10 and dist.get_rank() == 0:
            snapshot_path = os.path.join(saved_dir, '%d.pt' % (real_iter // 2000))
            torch.save({'state_dict': ffc_net.module.probe_net.state_dict(), 'lru': ffc_net.module.lru.state_dict(), 'fc': ffc_net.module.queue.cpu(), 'qp': ffc_net.module.queue_position_dict}, snapshot_path)

    return real_iter


def train(conf):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    dist.init_process_group('nccl', 'tcp://%s:%d' % (conf.ip, conf.port), world_size=conf.world_size, rank=conf.global_rank)
    instance_sampler = id_sampler = None
    instance_db = MultiLMDBDataset(conf.source_lmdb, conf.source_file)
    instance_sampler = DistributedSampler(instance_db)
    instance_loader = DataLoader(instance_db, conf.batch_size, False, instance_sampler, num_workers=8, pin_memory=False, drop_last=True)
    id_db = PairLMDBDatasetV2(conf.source_lmdb, conf.source_file)
    id_sampler = DistributedSampler(id_db)
    id_loader = DataLoader(id_db, conf.batch_size, False, id_sampler, num_workers=8, pin_memory=False, drop_last=True)
    logger.info('#class %d' % instance_db.num_class)

    net = FFC(conf.net_type, conf.feat_dim, conf.queue_size, conf.scale, conf.loss_type, conf.margin,
              conf.alpha, conf.neg_margin, conf.pretrained_model_path, instance_db.num_class).cuda(conf.local_rank)
    
    if conf.sync_bn:
        sync_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    else:
        sync_net = net
    ffc_net = ddp(sync_net, [conf.local_rank])
    # Only rank 0 has another validate process.
    optim_config = load_config('config/optim_config')
    optim, lr_scheduler = get_optim_scheduler(ffc_net.parameters(), optim_config)

    real_iter = 0
    logger.info('enter training procedure...')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(optim_config['epochs']):
        if optim_config['scheduler'] != 'ReduceLROnPlateau':
            lr_scheduler.update(epoch, 0.0)
        if instance_sampler is not None:
            instance_sampler.set_epoch(epoch)
        real_iter = train_one_epoch(id_loader, instance_loader, ffc_net, optim, epoch, conf, conf.saved_dir, real_iter, scaler, optim_config['scheduler'], lr_scheduler, optim_config['warmup'], optim_config['epochs'])

    id_db.close()
    instance_db.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='fast face classification.')
    conf.add_argument('ip', type=str)
    conf.add_argument('port', type=int)
    conf.add_argument('local_rank', type=int)
    conf.add_argument('global_rank', type=int)
    conf.add_argument('world_size', type=int)
    conf.add_argument('saved_dir', type=str, help='snapshot directory')

    conf.add_argument('--net_type', type=str, default='mobile', help='backbone type')
    conf.add_argument('--queue_size', type=int, default=7409, help='size of the queue.')
    conf.add_argument('--print_freq', type=int, default=1000, help='The print frequency for training state.')
    conf.add_argument('--pretrained_model_path', type=str, default='')
    conf.add_argument('--batch_size', type=int, default=256, help='batch size over all gpus.')
    conf.add_argument('--alpha', type=float, default=0.99, help='weight of moving_average')
    conf.add_argument('--loss_type', type=str, default='Arc', choices=['Arc', 'AM', 'SV'], help="loss type, can be softmax, am or arc")
    conf.add_argument('--margin', type=float, default=0.5, help='loss margin ')
    conf.add_argument('--scale', type=float, default=32.0, help='scaling parameter ')
    conf.add_argument('--neg_margin', type=float, default=0.25, help='scaling parameter ')
    conf.add_argument('--sync_bn', action='store_true', default=False)
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    args = conf.parse_args()
    logger.info('Start optimization.')
    
    args.source_lmdb = ['/path to msceleb.lmdb']
    args.source_file = ['/path to kv file']
    logger.info(args)
    train(args)
    logger.info('Optimization done!')

