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


def compute_roc(G, P, G_labels, P_labels, indices, flags, steps=20, low=-1.0, high=1.0 + 1e-5, num_bins=10000):
    """
    perform matrix multiplication between gallery and probe and get ROC. As G and P might be so large that it cannot
    be fed into gpu memory so we should do partitioned matrix multiplication step by step. I have to confess that rank1 might be higher than the result when strictly follows original protocol.But it may become too complicated so I choose the implementation below.
    :param G: gallery matrix, shape is #gallery x #dim
    :param P: probe matrix, shape is #probe x #dim
    :param G_labels: int32, gallery labels, shape is #gallerys x 1
    :param P_labels: int32, probe labels, shape is #probes x 1
    :param indices: used to index P and P_labels ONLY, it's numpy ndarray with data type int32
    :param flag: If flag is transpose, then G and P are the same, so are G_labels and P_labels, it means that
    we do matrix multiplication dot(G, G.T) since G = P and we just need to keep the upper triangular part of the result.
    Otherwise, we keep the whole result since G != P.
    :param steps: how many rows we wish to select from P to do multiplication with G
    :param low: low threshold
    :param high: high threshold
    :param num_bins: num of bins
    :return: positive and negative bins
    """
    my_rank = dist.get_rank()
    # logger.info('Rank %d start calculating matrix multiplication...' % my_rank)
    pos_bins = torch.zeros(num_bins).to(G.device)
    neg_bins = torch.zeros(num_bins).to(G.device)
    # logger.info('Rank %d start calculating matrix multiplication, device is %d, indices len is %d' % (my_rank, G.device.index, len(indices)))
    transpose = 1
    # enter main loop
    top1_sims_list = []

    for i in range(0, len(indices), steps):
        index = torch.from_numpy(indices[i:i + steps]).long()
        p = P[index] # shape is #steps x #dims
        pg = F.linear(p, G) # shape is #steps x #gallery
        binary_label = P_labels[index] == G_labels.T # shape is #steps x #gallery
        sim, index = torch.topk(pg, 2, 1)
        top1_sim = sim[:, 1] # exclude itself 
        top1_index = index[:, 1].view(-1, 1) # exclude itself so we use top2
        bool_hit_flag = torch.gather(binary_label, 1, top1_index).squeeze()
        hit_candidate_sim = top1_sim[bool_hit_flag]
        if hit_candidate_sim.numel() > 0:
            top1_sims_list.extend(hit_candidate_sim.tolist())
        pos_bins += torch.histc(pg[binary_label], num_bins, low, high)
        neg_bins += torch.histc(pg[~binary_label], num_bins, low, high)

        # logger.info('Rank %d, steps %d/%d' % (my_rank, i, len(indices)))
    dist.all_reduce(pos_bins)
    dist.all_reduce(neg_bins)
    num_probe = P.shape[0]
        # logger.info('#probe %d' % num_probe)
        # logger.info(pos_bins[-1])
    pos_bins[-1] -= num_probe
        # pos_bins /= 2
        # neg_bins /= 2
    # perform cumulative sum in reverse order
    num_positives = torch.sum(pos_bins).item()
    num_negatives = torch.sum(neg_bins).item()
    # logger.info('#negative samples %d, #positive samples %d' % (num_negatives, num_positives))
    far = (torch.flip(torch.cumsum(torch.flip(neg_bins.view(1, num_bins), [1]), 1), [1]).squeeze() / num_negatives).cpu().numpy()
    vr = (torch.flip(torch.cumsum(torch.flip(pos_bins.view(1, num_bins), [1]), 1), [1]).squeeze() / num_positives).cpu().numpy()
    # logger.info('find score...')

    vr6, idx6 = find_score(far, vr, 1e-6)
    vr7, idx7 = find_score(far, vr, 1e-7)
    vr8, idx8 = find_score(far, vr, 1e-8)
    vr9, idx9 = find_score(far, vr, 1e-9)
    # vr10, idx10 = find_score(far * 1000, vr, 1e-7)
    # logger.info('compute top1...')
    """
    top1_result_tensor = torch.zeros(4).to(G.device)

    top1_part_candidate = torch.FloatTensor(top1_sims_list)
    threshold_1e_6 = (high - low) / num_bins * idx6 + low
    top1_result_tensor[0] = torch.sum(top1_part_candidate >= threshold_1e_6).item() / num_probe

    threshold_1e_7 = (high - low) / num_bins * idx7 + low
    top1_result_tensor[1] = torch.sum(top1_part_candidate >= threshold_1e_7).item() / num_probe

    threshold_1e_8 = (high - low) / num_bins * idx8 + low
    top1_result_tensor[2] = torch.sum(top1_part_candidate >= threshold_1e_8).item() / num_probe

    threshold_1e_9 = (high - low) / num_bins * idx9 + low
    top1_result_tensor[3] = torch.sum(top1_part_candidate >= threshold_1e_9).item() / num_probe
    """


    table = PrettyTable(['far=1e-6', 'far=1e-7', 'far=1e-8', 'far=1e-9'])
    table.add_row([round(vr6 * 100, 2), round(vr7 * 100, 2), round(vr8 * 100, 2), round(vr9 * 100, 2)])

    if dist.get_rank() == 0:
        print(table)
        """
        tr1 = (high - low) / num_bins * idx6 + low
        tr2 = (high - low) / num_bins * idx7 + low
        tr3 = (high - low) / num_bins * idx8 + low
        logger.info('threshold = %f @far=1e-3' % tr1)
        logger.info('threshold = %f @far=1e-4' % tr2)
        logger.info('threshold = %f @far=1e-5' % tr3)
        """


def extract_features(net_type, pretrained_model_path, test_list, test_labels, device_list, feat_dim=512):
    assert len(device_list) >= 2
    extract_device = torch.device('cuda:%d' % device_list[0])
    checkpoint = torch.load(pretrained_model_path, map_location=extract_device)['state_dict']
    net = create_net(net_type)
    param = {}
    cl = len('backbone.')
    for k, v in checkpoint.items():
        # print(k, v.shape)
        # if 'backbone' in k:
        #    param[k[cl:]] = v
        # else:
        param[k] = v
    net.load_state_dict(param, strict=True)

    model = torch.nn.DataParallel(net, device_list, output_device=device_list[0]).to(extract_device)
    model.eval()
    batch_size = 128

    feat_device = torch.device('cuda:%d' % device_list[0])
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
            test_feat[gid_start:gid_end] = model(input_blob).to(feat_device)
            # if i % 100 == 0:
            #    logger.info('extract %d/%d images...' % (i * batch_size, len(test_list)))

    test_label = torch.from_numpy(np.array(test_labels)).to(test_feat.device)
    return test_feat, test_label
    # return test_feat, test_label_gpu


def process_worker(fn, fn_params, rank, world_size, port, ip='127.0.0.1'):
    dist.init_process_group(backend='nccl', init_method="tcp://%s:%d" % (ip, port),
                            world_size=world_size, rank=rank)
    # logger.info('start process %d/%d...' % (rank, world_size))
    fn(*fn_params)
    dist.destroy_process_group()


if __name__ == '__main__':
    net_type = 'r50'
    world_size = 4 # torch.cuda.device_count()
    start_device_id = 4
    port = int('your port number') 
    device_list = [4,5,6,7]
    model_path = sys.argv[1]
    feat_dim = 512
    gallery_dir = '/gallery/directory'
    probe_dir = '/probe/directory'
    dir2labels = {}
    gallery_list = []
    gallery_label = []
    gd = os.listdir(gallery_dir)
    next_label = 0
    for d in gd:
        sd = os.path.join(gallery_dir, d)
        if os.path.isdir(sd):
            if 'distractor' in d:
                for root_dir, sub_dirs, files in os.walk(sd):
                    for e in files:
                        ext = os.path.splitext(e)[1].lower()
                        if ext in ('.jpg', '.png'):
                            gallery_label.append(next_label)
                            gallery_list.append(os.path.join(root_dir, e))
                            has_image = True
            else:
                files = os.listdir(sd)
                has_image = False
                for e in files:
                    ext = os.path.splitext(e)[1].lower()
                    if ext in ('.jpg', '.png'):
                        gallery_label.append(next_label)
                        gallery_list.append(os.path.join(sd, e))
                        has_image = True
            if has_image:
                assert d not in dir2labels
                dir2labels[d] = next_label
                next_label += 1
    pd = os.listdir(probe_dir)
    probe_list = []
    probe_label = []
    for d in pd:
        sd = os.path.join(probe_dir, d)
        if os.path.isdir(sd):
            if d in dir2labels:
                label = dir2labels[d]
            else:
                label = next_label
                next_label += 1
            files = os.listdir(sd)
            has_image = False
            for e in files:
                ext = os.path.splitext(e)[1].lower()
                if ext in ('.jpg', '.png'):
                    probe_list.append(os.path.join(sd, e))
                    probe_label.append(label)
                    has_image = True
            if not has_image and d not in dir2labels:
                next_label -= 1
    mp.set_start_method('spawn')
    if not os.path.exists('fl.pkl'):
        gfeat, glabel = extract_features(net_type, model_path, gallery_list, gallery_label, device_list, feat_dim)
        pfeat, plabel = extract_features(net_type, model_path, probe_list, probe_label, device_list, feat_dim)
    else:
        logger.info('load parameters from disk...')
        param = torch.load('fl.pkl')
        gfeat = param['gfeat']
        glabel = param['glabel']
        pfeat = param['pfeat']
        plabel = param['plabel']
    # feat and label resides on device_list[0]
    glabel = glabel.view(-1, 1)
    plabel = plabel.view(-1, 1)
    indices_list = [[] for _ in range(world_size)]
    num_probes = pfeat.shape[0]
    state = 0
    left = num_probes % world_size
    unit = num_probes // world_size
    for i in range(0, num_probes - left, world_size):
        for j in range(world_size):
            if state == 0:
                indices_list[j].append(i + j)
            else:
                indices_list[j].append(i + world_size - 1 - j)
        state = (state + 1) % 2
    if left > 0:
        s = -1
        start_idx = unit * world_size
        for j in range(left):
            indices_list[s].append(start_idx + j)
            s -= 1

    P = Pool(world_size)
    for i in range(world_size):
        device_id = i + start_device_id
        if device_id != gfeat.device.index:
            gfeat_dev_i = gfeat.cuda(device_id)
            glabel_dev_i = glabel.cuda(device_id)
            gfeat_dev_i.share_memory_()
            glabel_dev_i.share_memory_()
            pfeat_dev_i = pfeat.cuda(device_id)
            plabel_dev_i = plabel.cuda(device_id)
            pfeat_dev_i.share_memory_()
            plabel_dev_i.share_memory_()
            P.apply_async(process_worker, args=(compute_roc, [gfeat_dev_i, pfeat_dev_i, glabel_dev_i, plabel_dev_i,
                                                              np.array(indices_list[i]), 1], i, world_size, port))
        else:
            gfeat.share_memory_()
            glabel.share_memory_()
            pfeat.share_memory_()
            plabel.share_memory_()
            P.apply_async(process_worker, args=(compute_roc, [gfeat, pfeat, glabel, plabel,
                                                              np.array(indices_list[i]), 1], i, world_size, port))

    P.close()
    P.join()
