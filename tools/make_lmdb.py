import os
import lmdb
import cv2
from datum_def_pb2 import Datum


class LMDB:
    def __init__(self, lmdb_path):
        self.map_size = 500 * 1024 * 1024
        self.env = lmdb.open(lmdb_path, map_size=self.map_size)
        self.kv = {}
        self.buf_size = 1000

    def put(self, k, v):
        if k in self.kv:
            print('%s is already in the db.' % k)
        else:
            self.kv[k] = v
            if len(self.kv) >= self.buf_size:
                self.commit()

    def commit(self):
        if len(self.kv) > 0:
            txn = self.env.begin(write=True)
            for k, v in self.kv.items():
                try:
                    txn.put(k, v)
                except lmdb.MapFullError:
                    txn.abort()
                    self.map_size = self.map_size * 3 // 2  # double map size and recommit
                    self.env.set_mapsize(self.map_size)
                    self.commit()
                    return
            try:
                txn.commit()
            except lmdb.MapFullError:
                txn.abort()
                self.map_size = self.map_size * 3 // 2
                self.env.set_mapsize(self.map_size)
                self.commit()
                return
            self.kv = {}
            txn.abort()

    def close(self):
        self.commit()
        self.env.close()


"""
assume that your images has the following structure
image_root
    sub_dir1
         --1.jpg
         --2.jpg
    sub_dir2
         --1.jpg
         --2.jpg
         --3.jpg
    ...
    sub_dirn
         --1.jpg
         --2.jpg
         --3.jpg
Where each subdirectory contains images from the same subject.
"""


def make_lmdb(image_src_dir, path_to_lmdb, db_name):
    dirs = os.listdir(image_src_dir)
    db = LMDB(path_to_lmdb)
    fo = open(os.path.join(path_to_lmdb, '%s_kv.txt' % db_name), 'w')
    next_label = 0
    for d in dirs:
        sub_dir = os.path.join(image_src_dir, d)
        if os.path.isdir(sub_dir):
            images = os.listdir(sub_dir)
            files = []
            for fn in images:
                ext = os.path.splitext(fn)[1].lower()
                if ext in ('.jpg', '.png', '.bmp', '.jpeg'):
                    files.append(os.path.join(sub_dir, fn))
            if len(files) > 0:
                for j, path in enumerate(files):
                    cv_img = cv2.imread(path)
                    key = '%s_%d_%d' % (db_name, next_label, j)
                    datum = Datum()
                    datum.data = cv2.imencode('.jpg', cv_img)[1].tobytes()
                    db.put(key.encode('utf-8'), datum.SerializeToString())
                    fo.write('%s %d\n' % (key, next_label))
                next_label += 1
    db.close()
    fo.close()
