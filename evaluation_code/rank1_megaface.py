import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import Pool
from prettytable import PrettyTable
import torch.multiprocessing as mp
import numpy as np
import os
import sys
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from resnet_def import create_net
import logging as logger
logger.basicConfig(format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
                   datefmt="%Y-%m-%d %H:%M:%S", level=logger.INFO)


class MegaFace(Dataset):
    def __init__(self, src_dir, file_list):
        self.file_list = file_list
        self.src_dir = src_dir

    def __len__(self,):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.src_dir, self.file_list[index]))
        img = cv2.resize(img, (224, 224))

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            return (buf.astype(np.float32) - 127.5) * 0.0078125
        else:
            img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
            return img.astype(np.float32)

def find_score(far, vr, target=1e-4):
    l = 0
    u = far.size - 1
    e = -1
    while u - l > 1:
        mid = (l + u) // 2
        if far[mid] == target:
            if target != 0:
                return vr[mid]
            else:
                e = mid
                break
        elif far[mid] < target:
            u = mid
        else:
            l = mid
    if target == 0:
        i = e
        while i >= 0:
            if far[i] != 0:
                break
            i -= 1
        if i >= 0:
            return vr[i + 1], i + 1
        else:
            return vr[u], u
    if target != 0 and far[l] / target >= 8:
        return 0.0, 0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point, u


def extract_features(net_type, pretrained_model_path, test_list, test_labels, device_list, feat_dim=512, plus_one=False):
    assert len(device_list) >= 2
    extract_device = torch.device('cuda:%d' % device_list[0])
    checkpoint = torch.load(pretrained_model_path, map_location=extract_device)['state_dict']
    net = create_net(net_type)
    param = {}
    cl = len('backbone.')
    for k, v in checkpoint.items():
        param[k] = v
    net.load_state_dict(param, strict=True)

    model = torch.nn.DataParallel(net, device_list, output_device=device_list[0]).to(extract_device)
    model.eval()
    batch_size = 512

    feat_device = torch.device('cuda:%d' % device_list[0])
    if plus_one:
        test_feat = torch.zeros((len(test_list) + 1, feat_dim), dtype=torch.float32).to(feat_device)
    else:
        test_feat = torch.zeros((len(test_list), feat_dim), dtype=torch.float32).to(feat_device)

    db = MegaFace('', test_list)
    db_loader = DataLoader(db, batch_size, False, num_workers=12, pin_memory=False, drop_last=False)

    gid_end = 0

    with torch.no_grad():
        bid = 0
        for i, img in enumerate(db_loader):
            input_blob = img.to(feat_device)
            gid_start = gid_end
            gid_end += img.shape[0]
            feat = model(input_blob).to(feat_device)
            test_feat[gid_start:gid_end] = feat
            if (i + 1) % 1000 == 0:
                logger.info('extract %d/%d images...' % ((i + 1) * batch_size, len(test_list)))

    if plus_one:
        test_label = torch.from_numpy(np.array(test_labels + [-1])).to(test_feat.device)
    else:
        test_label = torch.from_numpy(np.array(test_labels)).to(test_feat.device)
    return test_feat, test_label
    # return test_feat, test_label_gpu


if __name__ == '__main__':
    net_type = 'r50'
    world_size = 8
    start_device_id = 0
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    model_path = sys.argv[1]
    feat_dim = 512
    probe_dir = '/path/to/your/probe'
    distractor_dir = '/path/to/megaface/distractors'
    dir2labels = {}
    dir2range = {}
    probe_list = []
    probe_label = []
    pd = os.listdir(probe_dir)
    next_label = 0
    num_images = 0
    start_index = 0
    end_index = 0
    images_per_id = 0
    for d in pd:
        sd = os.path.join(probe_dir, d)
        if os.path.isdir(sd):
            files = os.listdir(sd)
            has_image = False
            for e in files:
                ext = os.path.splitext(e)[1].lower()
                if ext in ('.jpg', '.png'):
                    probe_label.append(next_label)
                    probe_list.append(os.path.join(sd, e))
                    has_image = True
                    num_images += 1
                    images_per_id += 1
            if has_image:
                assert d not in dir2labels
                assert d not in dir2range
                start_index = end_index
                end_index += images_per_id
                dir2range[d] = [start_index, end_index]
                dir2labels[d] = next_label
                images_per_id = 0
                next_label += 1
    distractor_label = []
    distractor_list = []
    for root, sub_dirs, files in os.walk(distractor_dir):
        for e in files:
            ext = os.path.splitext(e)[1].lower()
            if ext in ('.jpg', '.png'):
                distractor_label.append(-1)
                distractor_list.append(os.path.join(root, e))

    if not os.path.exists('fl.pkl'):
        pfeat, plabel = extract_features(net_type, model_path, probe_list, probe_label, device_list, feat_dim)
        dfeat, dlabel = extract_features(net_type, model_path, distractor_list, distractor_label, device_list, feat_dim, True)
    else:
        logger.info('load parameters from disk...')
        param = torch.load('fl.pkl')
        pfeat = param['pfeat']
        plabel = param['plabel']
        dfeat = param['dfeat']
        dlabel = param['dlabel']
        dir2range = param['dir2range']
    tp = 0
    total = 0
    gt = dfeat.shape[0] - 1
    steps = 10
    for d, se in dir2range.items():
        P = pfeat[se[0]: se[1]]
        M = P.shape[0]
        # logger.info('compute %s...' % d)
        inter_scores = []
        for i in range(0, M, steps):
            p = P[i:i + steps]
            m = p.shape[0]
            sims = F.linear(p, dfeat).view(m, -1).max(axis=1)[0].tolist()
            inter_scores.extend(sims)
        intra_sims = F.linear(P, P).cpu().numpy()
        total += (M - 1) * M
        for i in range(M):
            tp += np.sum(intra_sims[i] > inter_scores[i]) - 1

    acc = tp / (total * 1.0)
    logger.info('rank1@10^6 = %f' % (acc))

