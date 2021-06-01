from lru_python import LRU 
import lru_utils
import time

import random

if __name__ == '__main__':
    lru_p = LRU(1000000)
    lru_c = lru_utils.LRU(1000000)
    x = []
    for i in range(1000000):
        x.append(random.randint(0, 100))
    s = time.time()
    for e in x:
        lru_p.try_get(e)
    e = time.time()
    print('Python Time: %f ms.' % ((e - s) * 1000))
    s = time.time()
    for e in x:
        lru_c.try_get(e)
    e2 = time.time()
    print('C++ Time: %f ms.' % ((e2- s) * 1000))
    s = time.time()
    for i in range(5):
        lru_p.rollback_steps(256)
    e = time.time()
    print('Python rollback time: %f ms.' % (e - s))
    s = time.time()
    for i in range(5):
        lru_c.rollback_steps(256)
    e = time.time()
    print('C++ rollback time: %f ms.' % (e - s))
    keys = []
    values = []
    for k, v in lru_p:
        keys.append(k)
        values.append(v)
    keys_c = lru_c.keys()
    values_c = lru_c.values()
    flag = True
    for x, y in zip(keys, keys_c):
        if x != y:
            flag = False
            break
    if flag:
        print('Check passed!')
    else:
        print('Fatal error!')
    for x, y in zip(values, values_c):
        if x != y:
            flag = False
    if flag:
        print('Check passed!')
    else:
        print('Fatal error!')
