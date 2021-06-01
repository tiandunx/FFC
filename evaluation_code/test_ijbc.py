import numpy as np
import torch
import cv2
import os
from resnet_def import create_net
from prettytable import PrettyTable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import logging as logger
logger.basicConfig(format="%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s",
                   datefmt="%Y-%m-%d %H:%M:%S", level=logger.INFO)

class IJBC(Dataset):
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
            return vr[i + 1]
        else:
            return vr[u]
    if target != 0 and far[l] / target >= 8:
        return 0.0
    nearest_point = (target - far[l]) / (far[u] - far[l]) * (vr[u] - vr[l]) + vr[l]
    return nearest_point, l


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


def read_template_media_list():
    templates = []
    medias = []
    with open('test_data/ijbc/ijbc_face_tid_mid.txt', 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) > 0:
                g = l.split(' ')
                templates.append(int(g[1]))
                medias.append(int(g[2]))
    return np.array(templates), np.array(medias)


def read_template_pairs():
    t1 = []
    t2 = []
    label = []
    with open('test_data/ijbc/ijbc_template_pair_label.txt', 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) > 0:
                g = l.split(' ')
                t1.append(int(g[0]))
                t2.append(int(g[1]))
                label.append(int(g[2]))
    return np.array(t1), np.array(t2), np.array(label)


def get_image_feature(src_dir, net_type, pretrained_model_path, device_list, feat_dim=512):
    net = create_net(net_type)
    assert len(device_list) >= 2
    checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cuda:%d' % device_list[0]))['state_dict']
    param = {}
    cl = len('backbone.')
    for k, v in checkpoint.items():
        # if 'backbone' in k:
        #    param[k[cl:]] = v
        # else:
        param[k] = v
    net.load_state_dict(param, strict=True)
    model = torch.nn.DataParallel(net, device_list, device_list[0]).cuda(device_list[0])
    model.eval()
    image_list = []
    with open('test_data/ijbc/ijbc_name_5pts_score.txt', 'r') as fin:
        for line in fin:
            l = line.strip()
            if len(l) > 0:
                image_list.append(l.split(' ')[0])

    db = IJBC(src_dir, image_list)
    # logger.info('extract features...')
    batch_size = 1024
    db_loader = DataLoader(db, batch_size, False, num_workers=12, pin_memory=False, drop_last=False)
    # batch_images = np.zeros((batch_size, 3, 112, 112), dtype=np.float32)
    bid = 0
    batch_names = []
    eid = 0
    image_feat = np.zeros((len(image_list), feat_dim), dtype=np.float32)
    num_batches = 0
    with torch.no_grad():
        for batch_id, img in enumerate(db_loader): 
            input_blob = img.cuda(device_list[0])
            feature = model(input_blob).cpu().numpy()
            sid = eid
            eid += feature.shape[0]
            image_feat[sid:eid] = feature
            num_batches += 1
            if num_batches % 500 == 0:
                logger.info('process %d/%d images...' % (num_batches * batch_size, len(image_list)))
    return image_feat, image_list


def image2template_feature(img_feats, templates, medias):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 10000 == 0:
            logger.info('Finish Calculating {}/{} template features.'.format(count_template, len(unique_templates)))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates


def verification(template_norm_feats, unique_templates, p1, p2):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    # total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.squeeze()
        if c % 50 == 0:
            logger.info('Finish {}/{} pairs.'.format(c, len(sublists)))
    return score


def test_main(net_type, model_path, ijbc_src_dir, device_list):
    templates, medias = read_template_media_list()
    p1, p2, label = read_template_pairs()
    image_feats, image_list = get_image_feature(ijbc_src_dir, net_type, model_path, device_list)
    # with open('baseline_ijbc.pkl', 'wb') as fo:
    #    pickle.dump({'feat': image_feats, 'meta': image_list}, fo)
    template_norm_feats, unique_templates = image2template_feature(image_feats, templates, medias)
    scores = verification(template_norm_feats, unique_templates, p1, p2)
    far, vr, thresholds = compute_roc(scores, label)
    vr4, idx4 = find_score(far, vr, 1e-4)
    model_name = os.path.split(model_path)[1]
    # logger.error('%s vr = %f @far=1e-4, threshold is %f' % (model_name, vr4, thresholds[idx4]))
    vr5, idx5 = find_score(far, vr, 1e-5)
    vr6, idx6 = find_score(far, vr, 1e-6)
    return vr4, vr5, vr6


if __name__ == '__main__':
    net_type = 'r50'
    model_dir = '/path/to/your/snapshot'
    ijbc_src_dir = '/path/to/ijbc/image/directory'
    device_list = [0, 1, 2, 3, 4,5,6,7]
    models = os.listdir(model_dir)
    res = []
    for m in models:
        if m.endswith('.pt'):
            logger.info('evaluate %s...' % m)
            vr4, vr5, vr6 = test_main(net_type, os.path.join(model_dir, m), ijbc_src_dir, device_list)
            res.append([vr4, vr5, vr6, m])
    res.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    table = PrettyTable(['model_name', 'far=1e-4', 'far=1e-5', 'far=1e-6'])
    for e in res:
        table.add_row([e[-1], e[0], e[1], e[2]])
    print(table)

