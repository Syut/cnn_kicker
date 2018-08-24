#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import ddz_type


# 把字符串的牌型转换成数组，l行15列
def str2ary(cards_str, separator=','):
    arr = cards_str.split(separator) if cards_str.find(separator) > 0 else [cards_str]
    l = len(arr)
    ret = np.zeros([l, 15], dtype=np.int32)
    for i in range(l):
        for j in arr[i]:
            if j != 'P':
                ret[i][ddz_type.CARDS_CHAR2VALUE[j]] += 1
    ret = ret[0] if l == 1 else ret
    return ret


def ary2str(cards):
    buf = []
    for i in range(ddz_type.CARDS_TYPE_NUM):
        buf.extend([ddz_type.CARDS_VALUE2CHAR[i]] * cards[i])
    return ''.join(buf) if buf else 'P'


KICKER_TYPE = {
    '!': 0, '@': 1,  # 三带
    '(': 2, ')': 3,  # 四带
    '#': 4, '&': 5,  # 二联
    '$': 6, '*': 7,  # 三联
    '%': 8, '?': 9,  # 四联
    '^': 10  # 五联
}
# (length, width)
KICKER_PARAMS = {
    '!': (1, 1), '@': (1, 2),  # 三带
    '(': (2, 1), ')': (2, 2),  # 四带
    '#': (2, 1), '&': (2, 2),  # 二联
    '$': (3, 1), '*': (3, 2),  # 三联
    '%': (4, 1), '?': (4, 2),  # 四联
    '^': (5, 1)  # 五联
}
