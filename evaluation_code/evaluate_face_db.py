import numpy as np
import os
import cv2
from torch.utils.data import Dataset

__all__ = ['test_func', 'PairTestDataset']


class PairTestDataset(Dataset):
    def __init__(self, image_root_dir, pair_file):
        self.image_root_dir = image_root_dir
        self.test_list = []
        unique_files = set([])
        with open(pair_file, 'r') as fin:
            for line in fin:
                l = line.strip()
                if len(l) > 0:
                    g = l.split(' ')
                    x = g[0]
                    y = g[1]
                    unique_files.add(x)
                    unique_files.add(y)
                    label = int(g[2])
                    self.test_list.append([x, y, label])
        self.unique_len = len(unique_files)

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, index):
        path_x, path_y, label = self.test_list[index]
        x = cv2.imread(os.path.join(self.image_root_dir, path_x))
        y = cv2.imread(os.path.join(self.image_root_dir, path_y))
        # x = cv2.resize(x, (224, 224))
        # y = cv2.resize(y, (224, 224))
        if x.ndim == 2:
            buf = np.zeros((3, x.shape[0], x.shape[1]), dtype=np.uint8)
            buf[0] = x
            buf[1] = x
            buf[2] = x
            img_x = (buf.astype(np.float32) - 127.5) * 0.0078125
        else:
            img_x = ((x.transpose((2, 0, 1)) - 127.5) * 0.0078125).astype(np.float32)

        if y.ndim == 2:
            buf = np.zeros((3, y.shape[0], y.shape[1]), dtype=np.uint8)
            buf[0] = y
            buf[1] = y
            buf[2] = y
            img_y = (buf.astype(np.float32) - 127.5) * 0.0078125
        else:
            img_y = ((y.transpose((2, 0, 1)) - 127.5) * 0.0078125).astype(np.float32)
        return img_x, img_y, label


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

#
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


def test_agedb(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean


def test_cfp(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 700: (i + 1) * 700]
        test_score = score[i * 700: (i + 1) * 700]
        if i == 0:
            train_label = mask[700:]
            train_score = score[700:]
        elif i == 9:
            train_label = mask[:6300]
            train_score = score[:6300]
        else:
            train_label_1 = mask[:i * 700]
            train_label_2 = mask[(i + 1) * 700:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 700]
            train_score_2 = score[(i + 1) * 700:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(700):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 700
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean


def test_calfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean


def test_lfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean


def test_cplfw(mask, score):
    acc_list = np.zeros(10, np.float32)
    for i in range(10):
        test_label = mask[i * 600: (i + 1) * 600]
        test_score = score[i * 600: (i + 1) * 600]
        if i == 0:
            train_label = mask[600:]
            train_score = score[600:]
        elif i == 9:
            train_label = mask[:5400]
            train_score = score[:5400]
        else:
            train_label_1 = mask[:i * 600]
            train_label_2 = mask[(i + 1) * 600:]
            train_label = np.hstack([train_label_1, train_label_2])
            train_score_1 = score[:i * 600]
            train_score_2 = score[(i + 1) * 600:]
            train_score = np.hstack([train_score_1, train_score_2])

        far, vr, threshold = compute_roc(train_score, train_label)
        train_accuracy = (vr + 1 - far) / 2.0
        tr = threshold[np.argmax(train_accuracy)]
        num_right_samples = 0
        for j in range(600):
            if test_score[j] >= tr and test_label[j] == 1:
                num_right_samples += 1
            elif test_score[j] < tr and test_label[j] == 0:
                num_right_samples += 1
        acc_list[i] = num_right_samples * 1.0 / 600
    mean = np.mean(acc_list)
    std = np.std(acc_list) / np.sqrt(10)
    return mean

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] >= threshold)
    n = np.sum(scores[flags == 0] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, num_thresholds=1000):
    data_min = np.min(scores)
    data_max = np.max(scores)
    unit = (data_max - data_min) * 1.0 / num_thresholds
    thresholds = data_min + (data_max - data_min) * np.array(range(1, num_thresholds + 1)) / num_thresholds
    new_interval = thresholds - unit / 2.0 + 2e-6
    new_thresholds = np.append(new_interval, np.array(new_interval[-1] + unit))
    accuracys = np.zeros(new_thresholds.size)
    for i in range(new_thresholds.size):
        accuracys[i] = getAccuracy(scores, flags, new_thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(new_thresholds[max_index])
    return bestThreshold


def test_func(db_name):
    func = {'sllfw': test_lfw, 'calfw': test_calfw, 'cfp': test_cfp, 'agedb': test_agedb, 'cplfw': test_cplfw, 'lfw': test_lfw}
    if db_name not in func:
        raise Exception('Unknown %s' % db_name)
    return func[db_name]
