#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from ddz_type import *


class Hand(object):
    def __init__(self, type, tag, len, width, kicker=None):
        self.type = type
        self.tag = tag
        self.len = len
        self.width = width
        self.kicker = kicker or []

    def __bool__(self):
        return self.type != HandType.NONE

    def __gt__(self, other):
        if self.type == other.type:
            return self.tag > other.tag
        elif self.type == HandType.BOMB or self.type == HandType.NUKE:
            return other.type != HandType.NUKE
        elif other.type == HandType.BOMB or other.type == HandType.NUKE:
            return False
        else:
            raise ValueError('%r and %r have un-comparable types' % (repr(self), repr(other)))

    def __eq__(self, other):
        return self.type == other.type and self.tag == other.tag \
               and self.len == other.len and len(self.kicker) == len(other.kicker)

    def __lt__(self, other):
        return not (self > other or self == other)

    def __repr__(self):
        return '[TYPE: %s][TAG: %s(%d)][LEN: %d][WIDTH: %d][KICKER: %s]' % \
               (self.type, PokerType(self.tag), self.tag, self.len, self.width, str(self.kicker))

    def __str__(self):
        return ''.join([CARDS_VALUE2CHAR[x] for x in self.to_list()])

    def to_list(self, with_kicker=True):
        """
        like '334455' -> [0,0,1,1,2,2]
        :return: 
        """
        buf = list(range(self.tag, self.tag + self.len)) * self.width
        buf.sort()
        if with_kicker:
            buf.extend(self.kicker)
        return buf

    def to_array(self):
        """
        '334455' -> np.array([2,2,2,0,0,0,0,0,0,0,0,0,0,0,0])
        :return: 
        """
        buf = self.to_list()
        return [buf.count(x) for x in range(CARDS_TYPE_NUM)]

    def to_dict(self):
        """
        '3334' -> '333!'
        '33344' -> '333@'
        :return: 
        """
        key = ''.join([CARDS_VALUE2CHAR[x] for x in self.to_list(with_kicker=False)])
        if len(self.kicker) > 0:
            kicker_type = {HandType.TRIO_SOLO: {1: '!'},
                           HandType.TRIO_PAIR: {1: '@'},
                           HandType.DUAL_SOLO: {1: '('},
                           HandType.DUAL_PAIR: {1: ')'},
                           HandType.AIRPLANE_SOLO: {2: '#',
                                                    3: '$',
                                                    4: '%',
                                                    5: '^'},
                           HandType.AIRPLANE_PAIR: {2: '&',
                                                    3: '*',
                                                    4: '?'}
                           }
            return key + kicker_type[self.type][self.len]
        else:
            return key


ERR_HAND = Hand(HandType.NONE, tag=-1, len=0, width=0, kicker=None)


def parse_hand(hand):
    """
    :param hand: str, list or np.array
    :return: Hand that parsed
    """
    if isinstance(hand, str):
        hand_list = [CARDS_CHAR2VALUE[x] for x in list(hand)]
    elif isinstance(hand, list):
        hand_list = hand[:]
    elif isinstance(hand, np.ndarray):
        hand_list = []
        for i in range(CARDS_TYPE_NUM):
            if hand[i] > 0:
                hand_list.extend([i] * hand[i])
    else:
        raise TypeError('hand should be one of types: (str, list, np.array)')

    hand_list.sort()
    num = len(hand_list)
    if num == 1:  # 单
        return parse_solo(hand_list)
    elif num == 2:  # 对、王炸
        return parse_pair(hand_list) or parse_bomb(hand_list)
    elif num == 3:  # 三条
        return parse_trio(hand_list)
    elif num == 4:  # 3+1、炸弹
        return parse_trio_solo(hand_list) or parse_bomb(hand_list)
    elif num == 5:  # 3+2、单顺
        return parse_trio_pair(hand_list) or parse_solo_chain(hand_list)
    elif num == 6:  # 单顺、连对、3+3、4+1+1
        return parse_solo_chain(hand_list) or parse_pair_sisters(hand_list) or parse_airplane(hand_list) \
               or parse_dual_solo(hand_list)
    elif num in (7, 11, 13):  # 单顺
        return parse_solo_chain(hand_list)
    elif num == 8:  # 单顺、连对、3+3+1+1、4+2+2
        return parse_solo_chain(hand_list) or parse_pair_sisters(hand_list) or parse_airplane_solo(hand_list) \
               or parse_dual_pair(hand_list)
    elif num == 9:  # 单顺、3+3+3
        return parse_solo_chain(hand_list) or parse_airplane(hand_list)
    elif num == 10:  # 单顺、连对、3+3+2+2
        return parse_solo_chain(hand_list) or parse_pair_sisters(hand_list) or parse_airplane_pair(hand_list)
    elif num == 12:  # 单顺、连对、3+3+3+3、 (3 + 1) * 3
        return parse_solo_chain(hand_list) or parse_pair_sisters(hand_list) or parse_airplane(hand_list) \
               or parse_airplane_solo(hand_list)
    elif num == 14:  # 连对
        return parse_pair_sisters(hand_list)
    elif num == 15:  # 3*5、 (3 + 2) * 3
        return parse_airplane(hand_list) or parse_airplane_pair(hand_list)
    elif num == 16:  # 连对、(3 + 1) * 4
        return parse_pair_sisters(hand_list) or parse_airplane_solo(hand_list)
    elif num == 18:  # 连对、3 * 6
        return parse_pair_sisters(hand_list) or parse_airplane(hand_list)
    elif num == 20:  # 连对、(3 + 2) * 4、(3 + 1) * 5
        return parse_pair_sisters(hand_list) or parse_airplane_pair(hand_list) or parse_airplane_solo(hand_list)
    else:
        return ERR_HAND


def parse_solo(hand_list):
    if len(hand_list) != 1:
        return ERR_HAND
    return Hand(type=HandType.SOLO, tag=hand_list[0], len=1, width=1)


def parse_pair(hand_list):
    if len(hand_list) != 2 or hand_list[0] != hand_list[1]:
        return ERR_HAND
    return Hand(type=HandType.PAIR, tag=hand_list[0], len=1, width=2)


def parse_trio(hand_list):
    if len(hand_list) == 3 and hand_list[0] == hand_list[1] and hand_list[0] == hand_list[2]:
        return Hand(type=HandType.TRIO, tag=hand_list[0], len=1, width=3)
    return ERR_HAND


def parse_trio_solo(hand_list, is_sorted=True):
    if len(hand_list) != 4:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    if hand_list[0] != hand_list[1]:
        hand = parse_trio(hand_list[1:])
        if hand:
            hand.type = HandType.TRIO_SOLO
            hand.kicker = [hand_list[0]]
    elif hand_list[-1] != hand_list[-2]:
        hand = parse_trio(hand_list[:-1])
        if hand:
            hand.type = HandType.TRIO_SOLO
            hand.kicker = [hand_list[-1]]
    else:
        return ERR_HAND
    return hand


def parse_trio_pair(hand_list, is_sorted=True):
    if len(hand_list) != 5:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    if hand_list[0] == hand_list[1] and hand_list[1] != hand_list[2]:
        hand = parse_trio(hand_list[2:])
        if hand:
            hand.type = HandType.TRIO_PAIR
            hand.kicker = [hand_list[0]] * 2
    elif hand_list[-1] == hand_list[-2] and hand_list[-2] != hand_list[-3]:
        hand = parse_trio(hand_list[:-2])
        if hand:
            hand.type = HandType.TRIO_PAIR
            hand.kicker = [hand_list[-1]] * 2
    else:
        return ERR_HAND
    return hand


def parse_bomb(hand_list, is_sorted=True):
    if not is_sorted:
        hand_list.sort()

    if len(hand_list) == 4 and hand_list[0] == hand_list[1] \
            and hand_list[0] == hand_list[2] and hand_list[0] == hand_list[3]:
        return Hand(type=HandType.BOMB, tag=hand_list[0], len=1, width=4)
    elif len(hand_list) == 2 and hand_list[0] == CARDS_CHAR2VALUE['X'] and hand_list[1] == CARDS_CHAR2VALUE['D']:
        return Hand(type=HandType.NUKE, tag=hand_list[0], len=2, width=1)
    return ERR_HAND


def parse_dual_solo(hand_list, is_sorted=True):
    if len(hand_list) != 6:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    kicker = [x for x in hand_list if hand_list.count(x) == 1]
    if len(kicker) != 2 or kicker[0] == kicker[1] \
            or (CARDS_CHAR2VALUE['X'] in kicker and CARDS_CHAR2VALUE['D'] in kicker):
        return ERR_HAND

    hand_list_cpy = hand_list[:]
    for elem in kicker:
        hand_list_cpy.remove(elem)

    hand = parse_bomb(hand_list_cpy)
    if hand:
        hand.type = HandType.DUAL_SOLO
        hand.kicker = kicker
    return hand


def parse_dual_pair(hand_list, is_sorted=True):
    if len(hand_list) != 8:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    kicker = [x for x in hand_list if hand_list.count(x) == 2]
    if len(kicker) != 4 or kicker[0] == kicker[-1] \
            or CARDS_CHAR2VALUE['X'] in kicker or CARDS_CHAR2VALUE['D'] in kicker:
        return ERR_HAND

    hand_list_cpy = hand_list[:]
    for elem in kicker:
        hand_list_cpy.remove(elem)

    hand = parse_bomb(hand_list_cpy)
    if hand:
        hand.type = HandType.DUAL_PAIR
        hand.kicker = kicker
    return hand


def parse_solo_chain(hand_list, is_sorted=True):
    if len(hand_list) < 5 or len(hand_list) > 12 or hand_list[-1] > CARDS_CHAR2VALUE['A']:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    for i in range(1, len(hand_list)):
        if hand_list[i] != hand_list[i - 1] + 1:
            return ERR_HAND
    return Hand(type=HandType.SOLO_CHAIN, tag=hand_list[0], len=len(hand_list), width=1)


def parse_pair_sisters(hand_list, is_sorted=True):
    if not len(hand_list) in (6, 8, 10, 12, 14, 16, 18, 20) or hand_list[-1] > CARDS_CHAR2VALUE['A']:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    for i in range(1, len(hand_list) - 2, 2):
        if hand_list[i] != hand_list[i - 1] or hand_list[i] != hand_list[i + 1] - 1:
            return ERR_HAND
    if hand_list[-1] != hand_list[-2]:
        return ERR_HAND
    return Hand(type=HandType.PAIR_SISTERS, tag=hand_list[0], len=len(hand_list) // 2, width=2)


def parse_airplane(hand_list, is_sorted=True):
    if not len(hand_list) in (6, 9, 12, 15, 18) or hand_list[-1] > CARDS_CHAR2VALUE['A']:
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    for i in range(2, len(hand_list) - 3, 3):
        if hand_list[i] != hand_list[i - 1] or hand_list[i] != hand_list[i - 2] or hand_list[i] != hand_list[i + 1] - 1:
            return ERR_HAND
    if hand_list[-1] != hand_list[-2] or hand_list[-1] != hand_list[-3]:
        return ERR_HAND
    return Hand(type=HandType.AIRPLANE, tag=hand_list[0], len=len(hand_list) // 3, width=3)


def parse_airplane_solo(hand_list, is_sorted=True):
    if not len(hand_list) in (8, 12, 16, 20):
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    kicker = [x for x in hand_list if hand_list.count(x) == 1]
    hand_list_cpy = hand_list[:]
    for elem in kicker:
        hand_list_cpy.remove(elem)

    if len(hand_list_cpy) != len(kicker) * 3 \
            or (CARDS_CHAR2VALUE['X'] in kicker and CARDS_CHAR2VALUE['D'] in kicker):  # 不能同时带大小王
        return ERR_HAND

    hand = parse_airplane(hand_list_cpy)
    if hand:
        hand.type = HandType.AIRPLANE_SOLO
        hand.kicker = kicker
    return hand


def parse_airplane_pair(hand_list, is_sorted=True):
    if not len(hand_list) in (10, 15, 20):
        return ERR_HAND

    if not is_sorted:
        hand_list.sort()

    kicker = [x for x in hand_list if hand_list.count(x) == 2]
    hand_list_cpy = hand_list[:]
    for elem in kicker:
        hand_list_cpy.remove(elem)

    if len(hand_list_cpy) * 2 != len(kicker) * 3 or CARDS_CHAR2VALUE['X'] in kicker or CARDS_CHAR2VALUE['D'] in kicker:
        return ERR_HAND

    hand = parse_airplane(hand_list_cpy)
    if hand:
        hand.type = HandType.AIRPLANE_PAIR
        hand.kicker = kicker
    return hand


if __name__ == '__main__':
    import time

    beg = time.clock()
    hand = parse_hand('6789tjqk')
    end = time.clock()
    print(end - beg, repr(hand))
