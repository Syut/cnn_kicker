#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Defined the constant data and common structures in ddz.
# NOTICE: DO NOT MODIFY ANY VARIABLES IN THIS FILE.
#
# =======================================================

import enum

CARDS_VALUE2CHAR = {
    0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9', 7: 'T',
    8: 'J', 9: 'Q', 10: 'K', 11: 'A', 12: '2', 13: 'X', 14: 'D', 52: 'X', 53: 'D'
}

CARDS_CHAR2VALUE = {
    '3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '2': 12,
    'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, 'X': 13, 'D': 14,
    't': 7, 'j': 8, 'q': 9, 'k': 10, 'a': 11, 'x': 13, 'd': 14
}

CARDS_TOTAL_NUM = 54
CARDS_TYPE_NUM = 15
CARDS_TYPE_NUM_NO_JOKER = 13

BLACK_JOKER_ID = 52
RED_JOKER_ID = 53

SUITS_TYPE_NUM = 4

PLAYER_NUM = 3

FARMER_CARDS_NUM = 17
LORD_CARDS_NUM = 20
POT_CARDS_NUM = 3


@enum.unique
class HandType(enum.IntEnum):
    NONE = 0
    SOLO = 1
    PAIR = 2
    TRIO = 3
    TRIO_SOLO = 4
    TRIO_PAIR = 5
    BOMB = 6
    DUAL_SOLO = 7
    DUAL_PAIR = 8

    SOLO_CHAIN = 9
    PAIR_SISTERS = 10
    AIRPLANE = 11
    AIRPLANE_SOLO = 12
    AIRPLANE_PAIR = 13
    NUKE = 14


class PokerType(enum.IntEnum):
    NONE = -1
    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12
    BLACK_JOKER = 13
    RED_JOKER = 14


THREE = PokerType.THREE
FOUR = PokerType.FOUR
FIVE = PokerType.FIVE
SIX = PokerType.SIX
SEVEN = PokerType.SEVEN
EIGHT = PokerType.EIGHT
NINE = PokerType.NINE
TEN = PokerType.TEN
JACK = PokerType.JACK
QUEEN = PokerType.QUEEN
KING = PokerType.KING
ACE = PokerType.ACE
TWO = PokerType.TWO
BLACK_JOKER = PokerType.BLACK_JOKER
RED_JOKER = PokerType.RED_JOKER
