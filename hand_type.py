#!/usr/bin/python
# -*- coding: utf-8 -*-
import hand_parser as hp

char2label = {
    # 单
    'P': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13,
    'X': 14, 'D': 15,
    # 双
    '33': 16, '44': 17, '55': 18, '66': 19, '77': 20, '88': 21, '99': 22, 'TT': 23, 'JJ': 24, 'QQ': 25, 'KK': 26,
    'AA': 27, '22': 28,
    # 三
    '333': 29, '444': 30, '555': 31, '666': 32, '777': 33, '888': 34, '999': 35, 'TTT': 36, 'JJJ': 37, 'QQQ': 38,
    'KKK': 39, 'AAA': 40, '222': 41,
    # 五联单顺
    '34567': 42, '45678': 43, '56789': 44, '6789T': 45, '789TJ': 46, '89TJQ': 47, '9TJQK': 48, 'TJQKA': 49,
    # 六联单顺
    '345678': 50, '456789': 51, '56789T': 52, '6789TJ': 53, '789TJQ': 54, '89TJQK': 55, '9TJQKA': 56,
    # 七联单顺
    '3456789': 57, '456789T': 58, '56789TJ': 59, '6789TJQ': 60, '789TJQK': 61, '89TJQKA': 62,
    # 八联单顺
    '3456789T': 63, '456789TJ': 64, '56789TJQ': 65, '6789TJQK': 66, '789TJQKA': 67,
    # 九联单顺
    '3456789TJ': 68, '456789TJQ': 69, '56789TJQK': 70, '6789TJQKA': 71,
    # 十联单顺
    '3456789TJQ': 72, '456789TJQK': 73, '56789TJQKA': 74,
    # 十一联单顺
    '3456789TJQK': 75, '456789TJQKA': 76,
    # 十二联单顺
    '3456789TJQKA': 77,
    # 三联双顺
    '334455': 78, '445566': 79, '556677': 80, '667788': 81, '778899': 82, '8899TT': 83, '99TTJJ': 84, 'TTJJQQ': 85,
    'JJQQKK': 86, 'QQKKAA': 87,
    # 四联双顺
    '33445566': 88, '44556677': 89, '55667788': 90, '66778899': 91, '778899TT': 92, '8899TTJJ': 93, '99TTJJQQ': 94,
    'TTJJQQKK': 95, 'JJQQKKAA': 96,
    # 五联双顺
    '3344556677': 97, '4455667788': 98, '5566778899': 99, '66778899TT': 100, '778899TTJJ': 101, '8899TTJJQQ': 102,
    '99TTJJQQKK': 103, 'TTJJQQKKAA': 104,
    # 六联双顺
    '334455667788': 105, '445566778899': 106, '5566778899TT': 107, '66778899TTJJ': 108, '778899TTJJQQ': 109,
    '8899TTJJQQKK': 110, '99TTJJQQKKAA': 111,
    # 七联双顺
    '33445566778899': 112, '445566778899TT': 113, '5566778899TTJJ': 114, '66778899TTJJQQ': 115, '778899TTJJQQKK': 116,
    '8899TTJJQQKKAA': 117,
    # 八联双顺
    '33445566778899TT': 118, '445566778899TTJJ': 119, '5566778899TTJJQQ': 120, '66778899TTJJQQKK': 121,
    '778899TTJJQQKKAA': 122,
    # 九联双顺
    '33445566778899TTJJ': 123, '445566778899TTJJQQ': 124, '5566778899TTJJQQKK': 125, '66778899TTJJQQKKAA': 126,
    # 十联双顺
    '33445566778899TTJJQQ': 127, '445566778899TTJJQQKK': 128, '5566778899TTJJQQKKAA': 129,
    # 三带一
    '333!': 130, '444!': 131, '555!': 132, '666!': 133, '777!': 134, '888!': 135, '999!': 136, 'TTT!': 137, 'JJJ!': 138,
    'QQQ!': 139, 'KKK!': 140, 'AAA!': 141, '222!': 142,
    # 三带二
    '333@': 143, '444@': 144, '555@': 145, '666@': 146, '777@': 147, '888@': 148, '999@': 149, 'TTT@': 150, 'JJJ@': 151,
    'QQQ@': 152, 'KKK@': 153, 'AAA@': 154, '222@': 155,
    # 二联飞机带单
    '333444#': 156, '444555#': 157, '555666#': 158, '666777#': 159, '777888#': 160, '888999#': 161, '999TTT#': 162,
    'TTTJJJ#': 163, 'JJJQQQ#': 164, 'QQQKKK#': 165, 'KKKAAA#': 166,
    # 三联飞机带单
    '333444555$': 167, '444555666$': 168, '555666777$': 169, '666777888$': 170, '777888999$': 171, '888999TTT$': 172,
    '999TTTJJJ$': 173, 'TTTJJJQQQ$': 174, 'JJJQQQKKK$': 175, 'QQQKKKAAA$': 176,
    # 四联飞机带单
    '333444555666%': 177, '444555666777%': 178, '555666777888%': 179, '666777888999%': 180, '777888999TTT%': 181,
    '888999TTTJJJ%': 182, '999TTTJJJQQQ%': 183, 'TTTJJJQQQKKK%': 184, 'JJJQQQKKKAAA%': 185,
    # 五联飞机带单
    '333444555666777^': 186, '444555666777888^': 187, '555666777888999^': 188, '666777888999TTT^': 189,
    '777888999TTTJJJ^': 190, '888999TTTJJJQQQ^': 191, '999TTTJJJQQQKKK^': 192, 'TTTJJJQQQKKKAAA^': 193,
    # 二联飞机带双
    '333444&': 194, '444555&': 195, '555666&': 196, '666777&': 197, '777888&': 198, '888999&': 199, '999TTT&': 200,
    'TTTJJJ&': 201, 'JJJQQQ&': 202, 'QQQKKK&': 203, 'KKKAAA&': 204,
    # 三联飞机带双
    '333444555*': 205, '444555666*': 206, '555666777*': 207, '666777888*': 208, '777888999*': 209, '888999TTT*': 210,
    '999TTTJJJ*': 211, 'TTTJJJQQQ*': 212, 'JJJQQQKKK*': 213, 'QQQKKKAAA*': 214,
    # 四联飞机带双
    '333444555666?': 215, '444555666777?': 216, '555666777888?': 217, '666777888999?': 218, '777888999TTT?': 219,
    '888999TTTJJJ?': 220, '999TTTJJJQQQ?': 221, 'TTTJJJQQQKKK?': 222, 'JJJQQQKKKAAA?': 223,
    # 二联飞机
    '333444': 224, '444555': 225, '555666': 226, '666777': 227, '777888': 228, '888999': 229, '999TTT': 230,
    'TTTJJJ': 231, 'JJJQQQ': 232, 'QQQKKK': 233, 'KKKAAA': 234,
    # 三联飞机
    '333444555': 235, '444555666': 236, '555666777': 237, '666777888': 238, '777888999': 239, '888999TTT': 240,
    '999TTTJJJ': 241, 'TTTJJJQQQ': 242, 'JJJQQQKKK': 243, 'QQQKKKAAA': 244,
    # 四联飞机
    '333444555666': 245, '444555666777': 246, '555666777888': 247, '666777888999': 248, '777888999TTT': 249,
    '888999TTTJJJ': 250, '999TTTJJJQQQ': 251, 'TTTJJJQQQKKK': 252, 'JJJQQQKKKAAA': 253,
    # 五联飞机
    '333444555666777': 254, '444555666777888': 255, '555666777888999': 256, '666777888999TTT': 257,
    '777888999TTTJJJ': 258, '888999TTTJJJQQQ': 259, '999TTTJJJQQQKKK': 260, 'TTTJJJQQQKKKAAA': 261,
    # 六联飞机
    '333444555666777888': 262, '444555666777888999': 263, '555666777888999TTT': 264, '666777888999TTTJJJ': 265,
    '777888999TTTJJJQQQ': 266, '888999TTTJJJQQQKKK': 267, '999TTTJJJQQQKKKAAA': 268,
    # 四带单
    '3333(': 269, '4444(': 270, '5555(': 271, '6666(': 272, '7777(': 273, '8888(': 274, '9999(': 275, 'TTTT(': 276,
    'JJJJ(': 277, 'QQQQ(': 278, 'KKKK(': 279, 'AAAA(': 280, '2222(': 281,
    # 四带双
    '3333)': 282, '4444)': 283, '5555)': 284, '6666)': 285, '7777)': 286, '8888)': 287, '9999)': 288, 'TTTT)': 289,
    'JJJJ)': 290, 'QQQQ)': 291, 'KKKK)': 292, 'AAAA)': 293, '2222)': 294,
    # 炸
    '3333': 295, '4444': 296, '5555': 297, '6666': 298, '7777': 299, '8888': 300, '9999': 301, 'TTTT': 302, 'JJJJ': 303,
    'QQQQ': 304, 'KKKK': 305, 'AAAA': 306, '2222': 307,
    # 王炸
    'XD': 308,
    'ERROR': 309
}

label2char = dict(zip(char2label.values(), char2label.keys()))


def calculate_label(label_str):
    if label_str == 'P':
        return 0
    else:
        hand = hp.parse_hand(label_str)
        if hand.type == hp.HandType.NONE:
            return -1
        else:
            return char2label[hand.to_dict()]
