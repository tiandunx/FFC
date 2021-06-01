import torch
import numpy as np
from resnet_def import create_net
import os
from evaluate_face_db import *
from torch.utils.data import DataLoader


def list_mean(x):
    if len(x) == 0:
        return 0.0
    s = 0
    for e in x:
        s += e
    return round(s / len(x) * 100, 2)


def validate(net, data_loader, device_id):
    accs = []
    db_names = []
    with torch.no_grad():
        for db_name, db_loader in data_loader.items():
            scores = []
            labels = []
            for batch_idx, (image_x, image_y, label) in enumerate(db_loader):
                feat_x = net(image_x.cuda(device_id))
                feat_y = net(image_y.cuda(device_id))
                batch_scores = torch.sum(feat_x * feat_y, dim=1).cpu().numpy()
                batch_labels = label.numpy()
                scores.append(batch_scores)
                labels.append(batch_labels)
            acc = test_func(db_name)(np.hstack(labels), np.hstack(scores))
            accs.append(acc)
            db_names.append(db_name)
    return accs, db_names


if __name__ == '__main__':
    net_type = 'r50'
    model_dir = ''
    device_list = [0, 1, 2, 3, 4,5,6,7]
    net = torch.nn.DataParallel(create_net(net_type).cuda(device_list[0]), device_list)
    net.eval()
    models = os.listdir(model_dir)
    ms.sort()
    test_image_dir = 'root to test image source directory.'
    test_db = {'lfw': PairTestDataset(os.path.join(test_image_dir, 'lfw'), 'test_data/lfw/pair_LFW.txt'),
               'sllfw': PairTestDataset(os.path.join(test_image_dir, 'lfw'), 'test_data/lfw/pair_SLLFW.txt'),
               'agedb': PairTestDataset(os.path.join(test_image_dir, 'agedb30'), 'test_data/agedb30/pairs_agedb.txt'),
               'cfp': PairTestDataset(os.path.join(test_image_dir, 'cfp'), 'test_data/cfp/pairs_cfp.txt'),
               'calfw': PairTestDataset(os.path.join(test_image_dir, 'calfw/images_cropped'), 'test_data/calfw/pairs_calfw.txt'),
               'cplfw': PairTestDataset(os.path.join(test_image_dir, 'cplfw/images_cropped'), 'test_data/cplfw/pairs_cplfw.txt')}
    test_loaders = {}
    test_batch_size = 1000
    for k, v in test_db.items():
        test_loaders[k] = DataLoader(v, test_batch_size, False, num_workers=4, pin_memory=False, drop_last=False)
    best_acc = 0.0
    best_model = ''
    fo = open('%s.txt' % net_type, 'w')
    for m in ms:
        model_path = os.path.join(model_dir, m)
        params = torch.load(model_path)['state_dict']
        net.module.load_state_dict(params, strict=True)
        accs, db_names = validate(net, test_loaders, device_list[0])
        valid_log = '%s: ' % m
        for k, acc in enumerate(accs):
            valid_log += '%s %.2f, ' % (db_names[k], acc * 100.0)
        cur_mean_acc = list_mean(accs)
        valid_log += 'mean avg acc %.2f, best avg acc %.2f' % (cur_mean_acc, best_acc)
        print(valid_log)
        fo.write('%s\n' % valid_log)
        if cur_mean_acc > best_acc:
            best_acc = cur_mean_acc
            if best_model != '':
                os.remove(os.path.join(model_dir, best_model))
            best_model = m
        else:
            pass # os.remove(model_path)
    fo.close()
    print('best model is %s, best acc is %.2f' % (best_model, best_acc))
