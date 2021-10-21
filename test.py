#!/usr/bin/python3

from os.path import join;
from absl import flags, app;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_dataset import load_datasets;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('model', default = join('models', 'model_x2.h5'), help = 'model path');
  flags.DEFINE_string('image', default = None, help = 'image path');
  flags.DEFINE_integer('lr_size', default = 192, help = 'input size for low resolution image');
  flags.DEFINE_enum('scale', default = '2', enum_values = ['2','3','4'], help = 'train EDSR on which scale of DIV2K');
  flags.DEFINE_enum('method', default = 'bicubic', enum_values = ['area', 'bicubic', 'bilinear', 'gaussian', 'lanczos3', 'lanczos5', 'mitchellcubic', 'nearest'], help = 'downsample method for preprocess');

def main(unused_argv):
  model = tf.keras.models.load_model(FLAGS.model);
  img = cv2.imread(FLAGS.image);
  if img is None:
    (trainset_x2,testset_x2), (trainset_x3, testset_x3), (trainset_x4, testset_x4) = load_datasets((FLAGS.lr_size, FLAGS.lr_size), FLAGS.method);
    testset = testset_x2 if FLAGS.scale == 2 else (testset_x3 if FLAGS.scale == 3 else testset_x4);
    testset = testset.batch(1);
    for lr, hr in testset:
      outputs = model(lr);
      outputs = outputs + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
      large = tf.squeeze(outputs, axis = 0);
      large = large.numpy().astype(np.uint8)[:,:,::-1];
      cv2.imshow('sr', large);
      cv2.waitKey();
  else:
    inputs = tf.expand_dims(img, axis = 0); # inputs.shape = (1, h, w, 3)
    inputs = tf.cast(inputs, dtype = tf.float32) - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
    outputs = model(inputs);
    outputs = outputs + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
    large = tf.squeeze(outputs, axis = 0); # large.shape = (h, w, 3)
    large = large.numpy().astype(np.uint8)[:,:,::-1];
    cv2.imshow('sr', large);
    cv2.waitKey();

if __name__ == "__main__":
  add_options();
  app.run(main);
