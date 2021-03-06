#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

import trans_utils
import hand_type
import ddz_type
import hand_parser


def single(ary, n):
    """
    检测指定数量n的牌
    :param ary  : 1x15的数组
    :param n    : 数量
    :return     : one_hot形式
    """
    ret = np.zeros_like(ary)
    m = ary >= n
    ret[m] = 1
    return ret


def chain(ary, n, m):
    """
    # 检测指定长度的顺子
    :param ary  : 1x15的数组
    :param n    : 顺子长度
    :param m    : 1单2双3三
    :return     : 类似one_hot形式 34567 -> [1,1,1,1,1,0...]
    """
    ret = np.zeros_like(ary)
    num = 0
    end = 0
    for j in range(13):
        if ary[j] >= m and j != 12:
            num += 1
            if num >= n:
                end = j
        else:
            if num >= n:
                for k in range(num):
                    ret[end] = 1
                    end -= 1
            num = 0
    return ret


def samples_from_one_game(record, role=None):
    """
    从一副牌中挑出有带牌的情况（只适用于下面这种格式）
    :param record : 36J6JQ23QK23JQKAX;458QK23TK79TJA689;79TA478A4568457T2;59D;0,59JJJQQQ;0,KK;2,AA;0,22;0,33366;2,55444;0,XD;0,A;
    :param role   : 默认为None,学三家。否则学习对应位置，0地主1下家2上家
    :return       : 一副牌中所有的带牌信息
                    remains   : 当前手牌（打出这一手之前）
                    recorders : 记牌器（打牌过程中出过的牌）
                    hands     : 打出的牌
                    roles     : 角色
    """
    record = record[:-1] if record[-1] == ';' else record
    record_ary = record.split(';')

    cards = np.zeros((4, 15), dtype=np.int32)

    process = np.zeros((len(record_ary) - 4, 15), dtype=np.int32)
    process_role = np.zeros((len(record_ary) - 4), dtype=np.int32)

    remains = []
    recorders = []
    hands = []
    roles = []
    process_cursor = 0
    for i in range(len(record_ary)):
        if i < 4:
            # 手牌和底牌
            cards[i] = trans_utils.str2ary(record_ary[i])
        else:
            # 打牌过程
            one_hand = record_ary[i].split(',')
            process_role[i - 4] = int(one_hand[0])
            process[i - 4] = trans_utils.str2ary(one_hand[1])

            # 判断是否为带牌
            one_hand_type = hand_type.calculate_label(one_hand[1])
            if 130 <= one_hand_type <= 223 or 269 <= one_hand_type <= 294:
                if role is not None:
                    if role == process_role[i - 4]:
                        # hand
                        hands.append(one_hand[1])
                        # role
                        roles.append(role)
                        # remain
                        if len(remains) == 0:
                            remain = cards[role].copy()
                            remain = remain + cards[3] if role == 0 else remain
                        for j in range(process_cursor, i - 4):
                            if process_role[j] == role:
                                remain -= process[j]
                        remains.append(remain.copy())
                        # recorder
                        if len(recorders) == 0:
                            recorder = cards[3].copy()
                        for j in range(process_cursor, i - 4):
                            if process_role[j] == 0 and np.sum(cards[3]) > 0:
                                # check pot
                                hand_ary = process[j].copy()
                                hand_ary -= cards[3]
                                num = np.where(hand_ary < 0)[0]
                                cards[3] = np.zeros(15, dtype=np.int32)
                                for k in num:
                                    cards[3][k] = -hand_ary[k]
                                    hand_ary[k] += cards[3][k]
                                recorder += hand_ary
                            else:
                                recorder += process[j]
                        recorders.append(recorder.copy())
                        process_cursor = i - 4
                else:
                    # 学三家，没写
                    pass
    ret = list()
    for i in range(len(roles)):
        ret.append((remains[i], recorders[i], hands[i], roles[i]))
    return ret


def learning_sample(remain, recorder, hand, role):
    """
    获得训练用的样本（输入、输出）
    :param remain   : 当前手牌
    :param recorder : 记牌器
    :param hand     : 打出的一手 
    :param role     : 角色 
    :return:  input : 3 x 9 x 15
              label : one hot
    """
    hand = hand_parser.parse_hand(hand)
    main_hand = hand.to_dict()[:-1]
    cur_type = hand.to_dict()[-1]
    kicker_len, kicker_width = trans_utils.KICKER_PARAMS[cur_type]
    kicker_type = trans_utils.KICKER_TYPE[cur_type]
    cur_remain = remain - trans_utils.str2ary(main_hand)
    kickers = hand.kicker[::kicker_width]
    inputs = []
    labels = []
    used_labels = np.zeros(15, dtype=np.int32)
    for i in range(kicker_len):
        cur_main = main_hand[3 * i:3 * (i + 1)] if len(main_hand) % 3 == 0 else main_hand
        one_input = build_kicker_input(kicker_type, role, main_hand, cur_remain, kicker_width, kicker_len, cur_main, recorder,
                                       used_labels)
        label = kickers[i]
        used_labels[label] = 1
        label_one_hot = np.zeros(15, dtype=np.int32)
        label_one_hot[label] = 1
        inputs.append(one_input)
        labels.append(label_one_hot)
    return inputs, labels


def build_kicker_input(kicker_type, role, main_hand, remain, kicker_width, kicker_len, cur_main, recorder, used_labels):
    """
    拼装输入
    :param kicker_type  : 带牌类型
    :param role         : 角色
    :param main_hand    : 主牌，例如：3334 -> 333
    :param remain       : 剩余手牌（去掉了main_hand）
    :param kicker_width : 1单2双
    :param kicker_len   : 带牌数量
    :param cur_main     : 当前主牌,333444 -> 333
    :param recorder     : 记牌器,打牌过程中出现的牌及底牌
    :param labels       : 已经带出过的牌，还保留在手牌中，但不合法
    :return: 
    """
    # layer 1
    # line 0: kicker_type and role
    layer_1 = np.zeros((9, 15), dtype=np.int32)
    layer_1[0][kicker_type] = 1
    layer_1[0][role - 3] = 1
    # line 1: main_hand
    for i in main_hand[::3]:
        layer_1[1][ddz_type.CARDS_CHAR2VALUE[i]] = 1
    # line 2~5: one_hot_remain
    for i in range(len(remain)):
        if remain[i] > 0:
            layer_1[remain[i] + 1][i] = 1
    # line 6: legal_solo
    # line 7: legal_pair
    # line 8: empty
    legal_line = kicker_width + 5
    mask = remain >= kicker_width
    layer_1[legal_line][mask] = 1
    layer_1[legal_line] -= used_labels
    # layer 2
    # line 0: kicker_width and kicker_length
    layer_2 = np.zeros((9, 15), dtype=np.int32)
    layer_2[0][kicker_width - 1] = 1
    layer_2[0][kicker_len - 6] = 1
    # line 1: current main
    layer_2[1][ddz_type.CARDS_CHAR2VALUE[cur_main[0]]] = 1
    # line 2~5: one_hot_recorder
    for i in range(len(recorder)):
        if recorder[i] > 0:
            layer_2[recorder[i] + 1][i] = 1
    # line 6: solo appear
    # line 7: pair appear
    # line 8: empty
    mask = recorder > 0
    layer_2[legal_line][mask] = 1
    layer_2[legal_line] += layer_1[legal_line]
    mask = layer_2[legal_line] > 1
    temp = np.zeros(15, dtype=np.int32)
    temp[mask] = 1
    layer_2[legal_line] = temp
    # layer 3
    # legal out hand
    layer_3 = np.zeros((9, 15), dtype=np.int32)
    # line 0~1: empty
    # line 2: solo
    layer_3[2] = single(remain, 1)
    # line 3: pair
    layer_3[3] = single(remain, 2)
    # line 4: trio
    layer_3[4] = single(remain, 3)
    # line 5: boom
    layer_3[5] = single(remain, 4)
    if remain[13] == 1 and remain[14] == 1:
        layer_3[5][13] = 1
        layer_3[5][14] = 1
    # line 6: solo_chain
    layer_3[6] = chain(remain, 5, 1)
    # line 7: pair_chain
    layer_3[7] = chain(remain, 3, 2)
    # line 8: plane
    layer_3[8] = chain(remain, 2, 3)

    ret = np.vstack((layer_1, layer_2, layer_3))
    ret = ret.reshape(3, 9, 15)
    return ret


if __name__ == '__main__':
    game_str = '3444777TTTKKK69Q2;3555888JJJAAA47TK;3666999QQQ22258JA;3XD;' \
               '0,4443;1,5553;2,6663;0,777X;1,8887;2,9998;0,TTT9;1,JJJT;2,QQQJ;0,KKKQ;1,AAAK;2,222A;'
    game_str = '79KA2469467JA47XD;8JQ78TJA2TK2689TK;34535QK389Q35JQA2;6T5;' \
               '0,6667775T;0,J;1,K;2,2;0,X;0,K;1,A;0,2;0,99444;1,888TT;1,6;2,K;0,D;2,3333;2,4555;2,J;0,A;1,2;1,JJ;1,7;2,A;2,QQQ8;2,9;'
    game_str = '3444777TTTKKK49Q2;3555888JJJAAA67TK;3666999QQQ22258JA;3XD;' \
               '0,444477TT;'
    samples = samples_from_one_game(game_str, 0)
    for sample in samples:
        s_input, s_label = learning_sample(sample[0], sample[1], sample[2], sample[3])
    pass
