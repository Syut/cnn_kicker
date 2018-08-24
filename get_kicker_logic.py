import numpy as np
import trans_utils as tu
import hand_parser as hp


# 获得能出的所有牌
def get_all_hands(cards):
    ret = []
    for i in range(len(cards)):
        hand = []
        solo_hands = []
        pair_hands = []
        trio_hands = []
        bomb_hands = []
        plane_hands = []
        for a in range(13):
            # 单
            if cards[i][a] > 0:
                hand.append(str(hp.parse_hand([a])))
                # 双
                solo_hands.append([a])
            if cards[i][a] > 1:
                hand.append(str(hp.parse_hand([a, a])))
                pair_hands.append([a, a])
            # 三
            if cards[i][a] > 2:
                hand.append(str(hp.parse_hand([a, a, a])))
                trio_hands.append([a, a, a])
            # 四
            if cards[i][a] > 3:
                hand.append(str(hp.parse_hand([a, a, a, a])))
                bomb_hands.append([a, a, a, a])
        if cards[i][13] > 0:
            hand.append('X')
            solo_hands.append([13])
        if cards[i][14] > 0:
            hand.append('D')
            solo_hands.append([14])
        # 单顺
        for a in range(8):
            b = a
            num = 0
            while cards[i][b] != 0 and b < 12:
                num += 1
                if num > 4:
                    hand.append(str(hp.Hand(hp.HandType.SOLO_CHAIN, a, num, 1)))
                b += 1
        # 双顺
        for a in range(10):
            b = a
            num = 0
            while cards[i][b] > 1 and b < 12:
                num += 1
                if num > 2:
                    hand.append(str(hp.Hand(hp.HandType.PAIR_SISTERS, a, num, 2)))
                b += 1
        # 飞机
        for a in range(11):
            b = a
            num = 0
            while cards[i][b] > 2 and b < 12:
                num += 1
                if num > 1:
                    hand.append(str(hp.Hand(hp.HandType.AIRPLANE, a, num, 3)))
                    plane_hands.append(hp.Hand(hp.HandType.AIRPLANE, a, num, 3))
                b += 1

        if len(trio_hands) > 0:
            # 三带一
            for c in trio_hands:
                for d in solo_hands:
                    if c[0] != d[0]:
                        h = c.copy()
                        h.extend(d)
                        hand.append(str(hp.parse_hand(h)))
            # 三带二
            for c in trio_hands:
                for d in pair_hands:
                    if c[0] != d[0]:
                        h = c.copy()
                        h.extend(d)
                        hand.append(str(hp.parse_hand(h)))

        if len(bomb_hands) > 0:
            # 四带单
            for c in bomb_hands:
                for d in range(len(solo_hands)):
                    for e in range(d + 1, len(solo_hands)):
                        if c[0] != solo_hands[d][0] and c[0] != solo_hands[e][0] and not (
                                        solo_hands[d][0] == 13 and solo_hands[e][0] == 14):
                            h = c.copy()
                            h.extend(solo_hands[d])
                            h.extend(solo_hands[e])
                            hand.append(str(hp.parse_hand(h)))
            # 四带双
            for c in bomb_hands:
                for d in range(len(pair_hands)):
                    for e in range(d + 1, len(pair_hands)):
                        if c[0] != pair_hands[d][0] and c[0] != pair_hands[e][0]:
                            h = c.copy()
                            h.extend(pair_hands[d])
                            h.extend(pair_hands[e])
                            hand.append(str(hp.parse_hand(h)))

        if len(plane_hands) > 0:
            # 飞机带单
            for f in plane_hands:
                if len(solo_hands) >= f.len:
                    solo_kicker = solo_hands.copy()
                    g = f.to_list()[0::3]
                    for l in g:
                        solo_kicker.remove([l])
                    if f.len == 2:
                        for d in range(len(solo_kicker)):
                            for e in range(d + 1, len(solo_kicker)):
                                h = f.to_list().copy()
                                h.extend(solo_kicker[d])
                                h.extend(solo_kicker[e])
                                if not (13 in h and 14 in h):
                                    hand.append(str(hp.parse_hand(h)))
                    elif f.len == 3:
                        for d in range(len(solo_kicker)):
                            for e in range(d + 1, len(solo_kicker)):
                                for n in range(e + 1, len(solo_kicker)):
                                    h = f.to_list().copy()
                                    h.extend(solo_kicker[d])
                                    h.extend(solo_kicker[e])
                                    h.extend(solo_kicker[n])
                                    if not (13 in h and 14 in h):
                                        hand.append(str(hp.parse_hand(h)))
                    elif f.len == 4:
                        for d in range(len(solo_kicker)):
                            for e in range(d + 1, len(solo_kicker)):
                                for n in range(e + 1, len(solo_kicker)):
                                    for m in range(n + 1, len(solo_kicker)):
                                        h = f.to_list().copy()
                                        h.extend(solo_kicker[d])
                                        h.extend(solo_kicker[e])
                                        h.extend(solo_kicker[n])
                                        h.extend(solo_kicker[m])
                                        if not (13 in h and 14 in h):
                                            hand.append(str(hp.parse_hand(h)))
                    elif f.len == 5:
                        if not (13 in solo_hands and 14 in solo_hands) and len(solo_kicker) == 5:
                            h = f.to_list().copy()
                            for o in solo_kicker:
                                h.extend(o)
                            hand.append(str(hp.parse_hand(h)))
            # 飞机带双
            for f in plane_hands:
                if len(pair_hands) >= f.len:
                    pair_kicker = pair_hands.copy()
                    g = f.to_list()[0::3]
                    for l in g:
                        pair_kicker.remove([l, l])
                    if f.len == 2:
                        for d in range(len(pair_kicker)):
                            for e in range(d + 1, len(pair_kicker)):
                                h = f.to_list().copy()
                                h.extend(pair_kicker[d])
                                h.extend(pair_kicker[e])
                                hand.append(str(hp.parse_hand(h)))
                    elif f.len == 3:
                        for d in range(len(pair_kicker)):
                            for e in range(d + 1, len(pair_kicker)):
                                for n in range(e + 1, len(pair_kicker)):
                                    h = f.to_list().copy()
                                    h.extend(pair_kicker[d])
                                    h.extend(pair_kicker[e])
                                    h.extend(pair_kicker[n])
                                    hand.append(str(hp.parse_hand(h)))
                    elif f.len == 4:
                        if len(pair_kicker) == 4:
                            h = f.to_list().copy()
                            for o in pair_kicker:
                                h.extend(o)
                            hand.append(str(hp.parse_hand(h)))
        # 王炸
        if cards[i][13] == 1 and cards[i][14] == 1:
            hand.append('XD')
        ret.append(hand)
    return ret


def get_base_kickers(cards, kicker_type):
    """
    获取基本的带牌，原则：尽量不拆散手牌。方法：遍历所有合法带牌，如果带这张，计算剩下牌的所有合法打法，打法越多的说明越不重要，越应该被带。
    :param cards:除去带牌主体的剩余手牌 
    :param kicker_type: 0单1双
    :return: 候选带牌
    """
    # 如果不是只剩下2XD，不考虑
    if sum(cards[:-3]) > 0:
        c = []
        cards_index = []
        for i in range(len(cards)):
            if cards[i] > kicker_type:
                cp = cards.copy()
                cp[i] -= (kicker_type + 1)
                c.append(cp)
                cards_index.append(i)
        all_hands = get_all_hands(c)
        l = [len(x) for x in all_hands]
        filter_1 = np.where(l == np.max(l))[0]
        if len(filter_1) > 1:
            score1 = [0] * len(filter_1)
            for i in range(len(filter_1)):
                cp = cards.copy()
                cp[cards_index[filter_1[i]]] -= (kicker_type + 1)
                c1 = []
                for j in all_hands[filter_1[i]]:
                    cp1 = cp.copy()
                    cp1 -= tu.str2ary(j)
                    c1.append(cp1)
                all_hands1 = get_all_hands(c1)
                l1 = [len(x) for x in all_hands1]
                score1[i] = sum(l1)
            filter_2 = np.where(score1 == np.max(score1))[0]
            out = []
            for i in filter_2:
                out.append(cards_index[filter_1[i]])
        else:
            out = [cards_index[filter_1[0]]]
    else:
        out = []
        for i in range(12, 15):
            if cards[i] > 0:
                out.append(i)
                break
    return out


def logic_filter(base_kickers, cards, recorder, role):
    """
    带牌逻辑
    1.优先带出现过的牌
    2.如果是地主下家，留最小的准备送
    3.带最小的
    :param base_kickers: 经过组套(get_base_kickers)过滤过的候选带牌
    :param cards: 除去带牌主体的剩余手牌
    :param recorder: 出现过的牌
    :param role: 角色
    :return: 
    """
    last_filter = []
    for i in base_kickers:
        # 优先带出现过的牌
        if recorder[i] > 0:
            last_filter.append(i)
    if len(last_filter) == 0:
        last_filter = base_kickers

    if len(last_filter) > 1:
        # 地主下家留最小的
        if role == 1:
            out = last_filter[1]
        else:
            # 优先带最小的
            out = last_filter[0]
    else:
        out = last_filter[0]
    return out


def get_kicker(cards, pot, process, role, dict_hand):
    main_hand = dict_hand[:-1]
    kicker_type = len(dict_hand) - 4
    # kicker_type = 0 if '!' == dict_hand[-1] else 1
    remain = cards + pot if role == 0 else cards
    remain -= tu.str2ary(main_hand)
    recorder = pot.copy() if role == 0 else np.zeros(15, dtype=np.int32)
    for p in process:
        cur_role = int(p.split(',')[0])
        hand = tu.str2ary(p.split(',')[1])
        hand_pot = hand.copy()
        if cur_role == 0 and np.sum(pot) > 0:
            hand_pot -= pot
            num = np.where(hand_pot < 0)[0]
            pot = np.zeros(15, dtype=np.int32)
            for k in num:
                pot[k] = -hand_pot[k]
                hand_pot[k] += pot[k]
        if cur_role == role:
            remain -= hand
        recorder = recorder + hand_pot if role == 0 else recorder + hand
    base_kickers = get_base_kickers(cards, kicker_type)
    return logic_filter(base_kickers, cards, recorder, role)


if __name__ == '__main__':
    cards = [1, 1, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recorder = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    role = 2
    base_ks = get_base_kickers(cards, 0)
    best_kicker = logic_filter(base_ks, cards, recorder, role)
    print(best_kicker)
