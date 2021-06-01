import numpy as np
import cv2
import lmdb
from .datum_def_pb2 import Datum
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import random, sample
import torch
import torchvision.transforms as tf



class MultiLMDBDataset(Dataset):
    def __init__(self, source_lmdbs, source_files, feat_lmdbs=None, feat_files=None, transforms=None, return_feats=False):
        if (not isinstance(source_lmdbs, list)) and (not isinstance(source_lmdbs, tuple)):
            source_lmdbs = [source_lmdbs]
        if (not isinstance(source_files, list)) and (not isinstance(source_files, tuple)):
            source_files = [source_files]
        assert len(source_files) == len(source_lmdbs)
        assert len(source_lmdbs) > 0
        self.source_lmdbs = source_lmdbs
        self.train_list = []
        max_label = 0
        last_label = 0
        for db_id, file_path in enumerate(source_files):
            with open(file_path, 'r') as fin:
                for line in fin:
                    l = line.rstrip().lstrip()
                    if len(l) > 0:
                        items = l.split(' ')
                        self.train_list.append([items[0], db_id, int(items[1]) + last_label])
                        max_label = max(max_label, int(items[1]) + last_label)
            if max_label != last_label:
                max_label += 1
                last_label = max_label
        self.num_class = last_label

        self.transform = transforms
        self.feat_db = None
        self.return_feats = return_feats
        self.feat_lmdbs = feat_lmdbs
        self.feat_files = feat_files
        self.txns = None
        self.envs = None

        if transforms is not None:
            assert isinstance(transforms, list) or isinstance(transforms, tuple)
            assert len(self.transform) == len(source_lmdbs)

    def __len__(self):
        return len(self.train_list)

    def open_lmdb(self):
        self.txns = []
        self.envs = []
        for lmdb_path in self.source_lmdbs:
            self.envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=10, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False, buffers=False))
        if self.return_feats and (self.feat_lmdbs is not None and self.feat_files is not None):
            self.feat_db = KVDataset(self.feat_lmdbs, self.feat_files)

    def close(self):
        if self.txns is not None:
            for i in range(len(self.txns)):
                self.txns[i].abort()
                self.envs[i].close()
        if self.feat_db is not None:
            self.feat_db.close()

    def __getitem__(self, index):
        if self.envs is None:
            self.open_lmdb()
        lmdb_key, db_id, label = self.train_list[index][:3]
        datum = Datum()
        raw_byte = self.txns[db_id].get(lmdb_key.encode('utf-8'))
        datum.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        # img = cv2.resize(img, (224, 224)) # when compared to VFC, we should uncomment this line.
        if random() < 0.5:
            img = cv2.flip(img, 1)
        # func = tf.RandomErasing(0.1)

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            ret = tim
            if self.feat_db is not None:
                return ret, label, self.feat_db[lmdb_key]
            else:
                return ret, label, -1
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            ret = img
            if self.feat_db is not None:
                return ret, label, self.feat_db[lmdb_key]
            else:
                return ret, label, -1



class PairLMDBDataset(Dataset):
    def __init__(self, source_lmdb, pair_list, transforms=None):

        self.env = lmdb.open(source_lmdb, readonly=True, lock=False, max_readers=4, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

        self.train_list = pair_list

    def __len__(self):
        return len(self.train_list)

    def set_train_list(self, input_pairs):
        self.train_list = input_pairs

    def close(self):
        self.txn.abort()
        self.env.close()

    def __getitem__(self, index):
        key1, key2 = self.train_list[index]
        datum = Datum()
        raw_byte = self.txn.get(key1.encode('utf-8'))
        datum.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        p = random()
        if p < 0.5:
            img = cv2.flip(img, 1)
        func = tf.RandomErasing(0.05)

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            img_x = func(tim)
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            img_x = func(img)
        datum2 = Datum()
        raw_byte = self.txn.get(key2.encode('utf-8'))
        datum2.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum2.data, dtype=np.uint8), -1)
        p = random()
        if p < 0.5:
            img = cv2.flip(img, 1)

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            img_y = func(tim)
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            img_y = func(img)
        return img_x, img_y


class PairLMDBDatasetV2(Dataset):
    def __init__(self, source_lmdbs, source_files, exclude_id_set=None):
        if (not isinstance(source_lmdbs, list)) and (not isinstance(source_lmdbs, tuple)):
            source_lmdbs = [source_lmdbs]
        if (not isinstance(source_files, list)) and (not isinstance(source_files, tuple)):
            source_files = [source_files]
        assert len(source_files) == len(source_lmdbs)
        assert len(source_lmdbs) > 0
        self.envs = None
        self.txns = None
        self.source_lmdbs = source_lmdbs
        max_label = 0
        last_label = 0
        self.label2files = {}
        self.label_set = []
        for db_id, file_path in enumerate(source_files):
            with open(file_path, 'r') as fin:
                for line in fin:
                    l = line.strip()
                    if len(l) > 0:
                        items = l.split(' ')
                        the_label = int(items[1]) + last_label
                        if the_label not in self.label2files:
                            self.label2files[the_label] = [db_id, []]
                            self.label_set.append(the_label)
                        self.label2files[the_label][1].append(items[0])
                        max_label = max(max_label, the_label)
            max_label += 1
            last_label = max_label

    def __len__(self):
        return len(self.label_set)


    def open_lmdb(self):
        self.txns = []
        self.envs = []
        for lmdb_path in self.source_lmdbs:
            self.envs.append(lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=4, readahead=False, meminit=False))
            self.txns.append(self.envs[-1].begin(write=False))

    def close(self):
        if self.txns is not None:
            for i in range(len(self.txns)):
                self.txns[i].abort()
                self.envs[i].close()


    def __getitem__(self, index):
        if self.txns is None:
            self.open_lmdb()
        label = self.label_set[index]
        db_id, keys = self.label2files[label]
        if len(keys) >= 2:
            key1, key2 = sample(keys, 2)
        else:
            key1, key2 = keys[0], keys[0]
        datum = Datum()
        raw_byte = self.txns[db_id].get(key1.encode('utf-8'))
        datum.ParseFromString(raw_byte)
        img = cv2.imdecode(np.frombuffer(datum.data, dtype=np.uint8), -1)
        # img = cv2.resize(img, (224, 224))
        p = random()
        if p < 0.5:
            img = cv2.flip(img, 1)
        # func = tf.RandomErasing(0.1)

        if img.ndim == 2:
            buf = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.uint8)
            buf[0] = img
            buf[1] = img
            buf[2] = img
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)
            img_x = tim
        else:
            img = torch.from_numpy((img.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)
            img_x = img

        datum2 = Datum()
        raw_byte = self.txns[db_id].get(key2.encode('utf-8'))
        datum2.ParseFromString(raw_byte)
        img2 = cv2.imdecode(np.frombuffer(datum2.data, dtype=np.uint8), -1)
        # img2 = cv2.resize(img2, (224, 224))
        p = random()
        if p < 0.5:
            img2 = cv2.flip(img2, 1)

        if img2.ndim == 2:
            buf = np.zeros((3, img2.shape[0], img2.shape[1]), dtype=np.uint8)
            buf[0] = img2
            buf[1] = img2
            buf[2] = img2
            tim = torch.from_numpy((buf - 127.5).astype(np.float32) * 0.0078125)

            img_y = tim
        else:
            img2 = torch.from_numpy((img2.transpose((2, 0, 1)).astype(np.float32) - 127.5) * 0.0078125)

            img_y = img2
        return img_x, img_y, label


