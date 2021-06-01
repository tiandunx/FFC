import os
import torch
from resnet_def import create_net
import cv2
import numpy as np
from prettytable import PrettyTable
import pickle as cp
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging as logger
logger.basicConfig(format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
                   datefmt="%Y-%m-%d %H:%M:%S", level=logger.INFO)

class YTF(Dataset):
    def __init__(self, src_dir, file_list):
        self.file_list = file_list
        self.src_dir = src_dir

    def __len__(self,):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path, template_id = self.file_list[index]
        img = cv2.imread(os.path.join(self.src_dir, image_path))
        img = cv2.resize(img, (224, 224))
        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            return (buf.astype(np.float32) - 127.5) * 0.0078125, template_id
        else:
            img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
            return img.astype(np.float32), template_id


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
            return vr[i + 1]
        else:
            return vr[u]
    if target != 0 and far[l] / target >= 8:
        return 0.0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point


def compute_roc(score, label, num_thresholds=1000, show_sample_hist=False):
    pos_dist = score[label == 1]
    neg_dist = score[label == 0]

    num_pos_samples = pos_dist.size
    num_neg_samples = neg_dist.size
    # print('#pos pairs: %d, #neg pairs: %d' % (num_pos_samples, num_neg_samples))
    data_max = np.max(score)
    data_min = np.min(score)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    threshold = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = threshold - unit / 2.0 + 2e-6
    new_interval = np.append(new_interval, np.array(new_interval[-1] + unit))
    P = np.triu(np.ones(num_thresholds))

    pos_hist, dummy = np.histogram(pos_dist, new_interval)
    neg_hist, dummy2 = np.histogram(neg_dist, new_interval)
    pos_mat = pos_hist[:, np.newaxis]
    neg_mat = neg_hist[:, np.newaxis]

    assert pos_hist.size == neg_hist.size == num_thresholds
    far = np.dot(P, neg_mat) / num_neg_samples
    far = np.squeeze(far)
    vr = np.dot(P, pos_mat) / num_pos_samples
    vr = np.squeeze(vr)
    if show_sample_hist is False:
        return far, vr, threshold
    else:
        return far, vr, threshold, pos_hist, neg_hist


def test_lfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 250: (i + 1) * 250]
        test_score = score[i * 250: (i + 1) * 250]
        if i == 0:
            train_label = mask[250:]
            train_score = score[250:]
        elif i == 9:
            train_label = mask[:2250]
            train_score = score[:2250]
        else:
            train_label_1 = mask[:i * 250]
            train_label_2 = mask[(i + 1) * 250:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 250]
            train_score_2 = score[(i + 1) * 250:]
            train_score = np.hstack([train_score_1, train_score_2])

        # far, vr, threshold = compute_roc(train_score, train_label)
        # train_accuracy = (vr + 1 - far) / 2.0
        # tr = threshold[np.argmax(train_accuracy)]
        # determine the threshold that could reach the max accuracy
        max_acc = -1
        tr = 2.0
        for t in train_score:
            idx = train_score >= t
            pos_idx = train_label == 1
            neg_idx = train_label == 0
            correct_pos_samples = np.sum(idx & pos_idx)
            correct_neg_samples = np.sum((~idx) & neg_idx)
            cur_acc = (correct_neg_samples + correct_pos_samples) / train_score.shape[0]
            if cur_acc > max_acc:
                max_acc = cur_acc
                tr = t
        num_right_samples = 0
        for j in range(test_score.shape[0]):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / test_score.shape[0]
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean, std


def test_main(net_type, pretrained_model_path, ytf_src_dir, ytf_pairs, ytf_list, device_list):
    dir_set = set([])
    for p in ytf_pairs:
        dir_set.add(p[0])
        dir_set.add(p[1])

    assert len(device_list) >= 2
    device = torch.device('cuda:%d' % device_list[0])
    checkpoint = torch.load(pretrained_model_path, map_location=device)['state_dict']
    param = {}
    cl = len('backbone.')
    for k, v in checkpoint.items():
        # if 'backbone' in k:
        #    param[k[cl:]] = v
        # else:
        param[k] = v
    # print('#images: %d' % (len(filename2label.keys())))
    net = create_net(net_type)
    net.load_state_dict(param, strict=True)
    model = torch.nn.DataParallel(net, device_list, output_device=device_list[0]).to(device)
    model.eval()
    filename2feat = {}
    template2feat = {}
    batch_size = 512
    db = YTF(ytf_src_dir, ytf_list)
    feat_dim = -1
    db_loader = DataLoader(db, batch_size, False, num_workers=8, pin_memory=False, drop_last=False)
    with torch.no_grad():
        for i, (img, template_tuple) in enumerate(db_loader):
            img_gpu = img.to(device)
            feat = model(img_gpu)
            if feat_dim == -1:
                feat_dim = feat.shape[1]
            for j, template_id in enumerate(template_tuple):
                if template_id not in template2feat:
                    template2feat[template_id] = []
                    filename2feat[template_id] = np.zeros(feat_dim, dtype=np.float32)
                template2feat[template_id].append(feat[j].view(1, -1))
            if (i + 1) % 200 == 0:
                logger.info('extract %d/%d batches...' % (i + 1, len(db_loader)))
    logger.info('transfer features...')
    for k in filename2feat.keys():
        filename2feat[k] = F.normalize(torch.mean(torch.cat(template2feat[k], dim=0), dim=0).view(1, -1)).cpu().numpy().squeeze()
    # with open('lfw_feat.dat', 'wb') as fout:
    #    cp.dump(filename2feat, fout, 1)
    # print('running lfw evaluation procedure...')
    score_list = []
    label_list = []
    for pairs in ytf_pairs:
        feat1 = filename2feat[pairs[0]]
        feat2 = filename2feat[pairs[1]]
        dist = np.dot(feat1, feat2) / np.sqrt(np.dot(feat1, feat1) * np.dot(feat2, feat2))
        score_list.append(dist)
        label_list.append(pairs[2])
    score = np.array(score_list)
    label = np.array(label_list)
    ytf_acc, ytf_std = test_lfw(label, score)
    logger.info('%s acc is %f' % (pretrained_model_path, ytf_acc))
    return ytf_acc, ytf_std


if __name__ == '__main__':
    net_type = 'r50'
    # model_dir = '/home/tiandu.ws/experiments/ffc_ddp/mobile_insloader_ms'
    model_dir = 'r50_ms_0.01id'
    ytf_src_dir = '/path/to/ytf/'
    ytf_pairs = []
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    with open('test_data/ytf/pairs_ytf.txt', 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) > 0:
                g = l.split(' ')
                ytf_pairs.append([g[0], g[1], int(g[2])])
    ytf_list = []
    with open('test_data/ytf/ytf_list.txt', 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) > 0:
                g = l.split(' ')
                ytf_list.append([g[0], g[1]])
    logger.info('#ytf list = %d' % (len(ytf_list)))

    models = os.listdir(model_dir)
    res = []
    for m in models:
        if m.endswith('.pt'):
            acc, std = test_main(net_type, os.path.join(model_dir, m), ytf_src_dir,
                                 ytf_pairs, ytf_list, device_list)
            res.append([acc, std, m])
    res.sort(key=lambda x: (-x[0], x[1]))
    table = PrettyTable(['model_name', 'ytf acc', 'ytf std'])
    for e in res:
        table.add_row([e[-1], e[0], e[1]])
    print(table)
