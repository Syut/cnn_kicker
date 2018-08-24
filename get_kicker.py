#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import cnn_kicker_model
import trans_utils
import hand_type
from learning_input_trans import build_kicker_input


class GetKicker(object):
    def __init__(self, model_path='./kicker_model', session_config=None):
        self._tf_session_config = session_config
        self.model_path = model_path
        self.saver = None

        self._init_graph()
        self._init_session()
        self._load_model()

    def _init_graph(self):
        # restore graph from meta
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            x = tf.placeholder(tf.float32, [3, 9, 15], name='x_input')
            # Store layers weight & bias
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=0.05)),
                'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.05)),
                'wc3': tf.Variable(tf.random_normal([3, 1, 32, 64], stddev=0.05)),
                'wc4': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
                'wc5': tf.Variable(tf.random_normal([3, 1, 64, 64], stddev=0.05)),
                # fully connected
                'wd1': tf.Variable(tf.random_normal([15 * 64, 512], stddev=0.04)),
                # 512 inputs, 309 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([512, 15], stddev=1 / 512.0))
            }
            biases = {
                'bc1': tf.Variable(tf.random_normal([16])),
                'bc2': tf.Variable(tf.random_normal([32])),
                'bc3': tf.Variable(tf.random_normal([64])),
                'bc4': tf.Variable(tf.random_normal([64])),
                'bc5': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([512])),
                'out': tf.Variable(tf.random_normal([15]))
            }
            restore_var = {
                'wc1': weights['wc1'],
                'wc2': weights['wc2'],
                'wc3': weights['wc3'],
                'wc4': weights['wc4'],
                'wc5': weights['wc5'],
                'wd1': weights['wd1'],
                'wout': weights['out'],
                'bc1': biases['bc1'],
                'bc2': biases['bc2'],
                'bc3': biases['bc3'],
                'bc4': biases['bc4'],
                'bc5': biases['bc5'],
                'bd1': biases['bd1'],
                'bout': biases['out']
            }

            # Construct model
            pred = cnn_kicker_model.conv_net(x, weights, biases, 1, False)
            pred_top = tf.nn.top_k(tf.nn.softmax(pred), k=5)
            tf.add_to_collection('pred', pred_top)
            # tf.add_to_collection('pred', tf.nn.softmax(pred))

            sc = tf.get_collection("scale")
            bt = tf.get_collection("beta")
            pm = tf.get_collection("pop_mean")
            pv = tf.get_collection("pop_var")
            for i in range(len(sc)):
                restore_var['scale' + str(i)] = sc[i]
                restore_var['beta' + str(i)] = bt[i]
                restore_var['pop_mean' + str(i)] = pm[i]
                restore_var['pop_var' + str(i)] = pv[i]

            self.saver = tf.train.Saver(restore_var)

    def _init_session(self):
        if self._tf_session_config is None:
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.05
            config.gpu_options.allow_growth = True
            self._tf_session_config = config

        self._tf_session = tf.Session(graph=self._tf_graph, config=self._tf_session_config)

    def _load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if self.saver is not None:
            self.saver.restore(self._tf_session, ckpt.model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find model! path=', self.model_path)

    def get_kicker(self, cards, pot, process, role, hand):
        main_hand = hand[:-1]
        cur_type = hand[-1]
        kicker_len, kicker_width = hand_type.KICKER_PARAMS[cur_type]
        kicker_type = hand_type.KICKER_TYPE[cur_type]
        ret_kickers = np.zeros(15, dtype=np.int32)
        remain = cards + pot if role == 0 else cards
        remain -= trans_utils.str2ary(main_hand)
        recorder = pot.copy() if role == 0 else np.zeros(15, dtype=np.int32)
        for p in process:
            cur_role = int(p.split(',')[0])
            hand = trans_utils.str2ary(p.split(',')[1])
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
        for i in range(kicker_len):
            cur_main = main_hand[3 * i:3 * (i + 1)] if len(main_hand) % 3 == 0 else main_hand
            x_input = build_kicker_input(kicker_type, role, main_hand, remain, kicker_width, kicker_len, cur_main, recorder,
                                         ret_kickers)
            all_kickers = self._tf_session.run(self._tf_graph.get_collection('pred'),
                                               feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input})
            kicker = all_kickers[0][1][0][0]
            ret_kickers[kicker] += 1
        ret = trans_utils.ary2str(ret_kickers * kicker_width)
        return ret


if __name__ == '__main__':
    import hand_parser

    # 34457888999QQKKAD;3346789TJJQQKAA22;345556667TTJKA22X;7TJ; lord=2; point=2; learn=0; 0,55566634;1,88899935;1,44;2,QQ;1,KK;0,22;0,TTT77;0,JJ;2,AA;2,6789T;2,33;1,QQ;1,7;2,K;0,A;1,D;1,A; [2, 2, -4]
    game_str = '345556667TTJKA22X;34457888999QQKKAD;3346789TJJQQKAA22;7TJ;0,55566634;1,88899935'
    ary = game_str.split(';')
    last_hand = ary.pop(-1)
    role = int(last_hand.split(',')[0])
    dict_hand = hand_parser.parse_hand(last_hand.split(',')[1]).to_dict()
    cards = trans_utils.str2ary(ary[role])
    pot = trans_utils.str2ary(ary[3])
    process = ary[4:]
    env = GetKicker()
    out = env.get_kicker(cards, pot, process, role, dict_hand)
    print(out)
