# coding=utf-8

import os
import tensorflow.compat.v1 as tf
from PIL import Image
import struct
import sys


def creat_tf(args):

    writer = tf.python_io.TFRecordWriter(args.train_tfrecords)
    img_org_path = args.input_image
    img_half_path = args.input_image_half
    img_quater_path = args.input_image_quater
    if os.path.isdir(img_org_path):
        img_list = os.listdir(img_org_path)
        for img_name in img_list:
            img_org = Image.open(img_org_path + '/' + img_name)
            if os.path.exists(img_half_path + '/' + img_name) and os.path.exists(img_quater_path + '/' + img_name):
                img_half = Image.open(img_half_path + '/' + img_name)
                img_quater = Image.open(img_quater_path + '/' + img_name)
            else:
                print(img_name)
                continue
            img_raw_org = img_org.tobytes()
            img_raw_half = img_half.tobytes()
            img_raw_quater = img_quater.tobytes()
            # length = len(img_raw)
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw_org': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_org])),
                'img_raw_half': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_half])),
                'img_raw_quater': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw_quater]))
            }))
            writer.write(example.SerializeToString())
                # print(img_name)
    writer.close()


def read_example():
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        image = image[0]
        label = example.features.feature['label'].int64_list.value

        print(label)



if __name__ == '__main__':
    parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_tfrecords', type=str, default='./clic-train-spatial.tfrecords', help="Tfrecords output path.")
    parser.add_argument('--input_image', type=str, default='./clic-train-512', help="Training image folder of base layer.")
    parser.add_argument('--input_image_half', type=str, default='./clic-train-half', help="Training image folder of e1 layer. 1/2 width and height of base layer image.")
    parser.add_argument('--input_image_quater', type=str, default='./clic-train-quater', help="Training image folder of e2 layer. 1/4 width and height of base layer image.")

    creat_tf(args)

