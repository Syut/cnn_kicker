#!/usr/bin/python
# -*- coding: utf-8 -*-
import ddz_type


def pre_trans(game):
    '''
    第二种输入形式，处理成第一种输入形式  第一种，默认0号位地主。底牌都是给0号位的。第二种，带花色，第一个出牌的是地主。
    第一种：A4J5TT28TQDA9QJ44;5287KK8K5676K22T7;39J93937A8QQ6X356;4JA;0,58TTTJJJ;0,2;2,X;0,D;0,QQ;1,22;1,T;0,4444;1,KKKK;1,55667788;1,7;0,9;1,2;
    第二种：4_7_10_11_14_15_21_23_25_32_34_38_43_47_48_50_52;0_8_9_12_13_19_20_24_26_28_30_33_36_39_42_44_49;2_3_5_6_16_17_18_22_27_29_31_35_37_40_41_51_53;1_45_46;Seat1:1_28_42_30_44_19|Seat2:|Seat0:|Seat1:20_33|Seat2:22_35|Seat0:23_10|Seat1:|Seat2:|Seat0:14|Seat1:36|Seat2:37|Seat0:|Seat1:0_13_26_39|Seat2:|Seat0:|Seat1:45_46_8_9_49_24|Seat2:|Seat0:|Seat1:12
    '''
    game_s = game.split(';')
    process = game_s[4].split('|')
    a = 0
    while process[a].split(':')[1] == '':
        a += 1
    lord = int(process[a].split(':')[0][-1])
    pos = (3 - lord) % 3
    game_str = color_card2str(game_s[lord].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    lord += 1
    game_str += color_card2str(game_s[lord % 3].split('_')) + ';'
    game_str += color_card2str(game_s[3].split('_')) + ';'
    r = 0 - a
    for i in process:
        temp = i.split(':')[1]
        if temp != '':
            game_str += str(r) + ',' + color_card2str(temp.split('_')) + ';'
        r = (r + 1) % 3
    return game_str, pos


# 带花色的牌，转化成字符串，非通用方法
def color_card2str(color_card_str):
    ret = ''
    for i in color_card_str:
        value = int(i)
        if value == 52:
            ret += 'X'
        elif value == 53:
            ret += 'D'
        else:
            ret += ddz_type.CARDS_VALUE2CHAR[value % 13]
    return ret
