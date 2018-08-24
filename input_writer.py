import os
import numpy as np
import tensorflow as tf
import multiprocessing

from read_records import pre_trans
from learning_input_trans import samples_from_one_game, learning_sample


class MyProcess(multiprocessing.Process):
    def __init__(self, process_id, name, counter, lock, base_path='D:/data_source'):
        multiprocessing.Process.__init__(self)
        self.process_id = process_id
        self.name = name
        self.counter = counter
        self.lock = lock
        self.base_path = base_path

    def run(self):
        print("开始进程：" + self.name)
        write_records(self.counter, self.lock, self.base_path)
        print("退出进程：" + self.name)


def build_bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def build_example(x, y):
    example = tf.train.Example(features=tf.train.Features(feature={
        'input': build_bytes_feature(x),
        'label': build_bytes_feature(y)
    }))
    return example


# base_path:牌谱源文件及保存文件的基础路径
def write_records(cnt, lock, base_path='D:/data_source'):
    lock.acquire()
    train_path = '/kicker/training'
    test_path = '/kicker/test'
    if not os.path.exists(base_path + train_path):
        os.makedirs(base_path + train_path)
    if not os.path.exists(base_path + test_path):
        os.makedirs(base_path + test_path)
    lock.release()
    train_file_pattern = base_path + train_path + '/kicker_training_%.4d.tfrecords'
    test_file_pattern = base_path + test_path + '/kicker_test_%.4d.tfrecords'
    file_src = base_path + '/ddz%.d.txt'
    LINE_NUM = 100000
    train_file_no = cnt * 1000 + 1
    train_line_cnt = 0
    test_file_no = cnt * 1000 + 1
    test_line_cnt = 0

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.01
    with tf.Session(config=config):
        with open(file_src % cnt) as fp:
            train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)
            test_writer = tf.python_io.TFRecordWriter(test_file_pattern % test_file_no)
            for line in fp.readlines():
                record, role = pre_trans(line)
                # print(record, role)
                samples = samples_from_one_game(record, role)
                for sample in samples:
                    s_input, s_label = learning_sample(sample[0], sample[1], sample[2], sample[3])
                    for i in range(len(s_input)):
                        if (train_line_cnt + test_line_cnt) % 10 > 0:
                            train_writer.write(build_example(x=np.array(s_input[i], dtype=np.uint8).tobytes(),
                                                             y=np.array(s_label[i], dtype=np.uint8).tobytes()).SerializeToString())
                            train_line_cnt += 1
                            if train_line_cnt >= LINE_NUM:  # 文件结束条件
                                train_writer.close()
                                train_line_cnt = 0
                                train_file_no += 1
                                train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)
                        else:
                            test_writer.write(build_example(x=np.array(s_input[i], dtype=np.uint8).tobytes(),
                                                            y=np.array(s_label[i], dtype=np.uint8).tobytes()).SerializeToString())
                            test_line_cnt += 1
                            if test_line_cnt >= LINE_NUM:  # 文件结束条件
                                test_writer.close()
                                test_line_cnt = 0
                                test_file_no += 1
                                test_writer = tf.python_io.TFRecordWriter(test_file_pattern % test_file_no)
            train_writer.close()
            test_writer.close()


if __name__ == '__main__':
    # 创建新线程
    process_list = []
    lock = multiprocessing.Lock()
    for i in range(5, 9):
        t = MyProcess(i, "Process-" + str(i), i, lock)
        t.start()
        process_list.append(t)

    for process in process_list:
        process.join()
