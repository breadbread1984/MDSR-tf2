#!/usr/bin/python3

from absl import app, flags;
import tensorflow as tf;
import tensorflow_datasets as tfds;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_bool('test', default = False, help = 'test dataset input pipeline');

def download():
  div2k_x2_builder = tfds.builder('div2k/bicubic_x2');
  div2k_x3_builder = tfds.builder('div2k/bicubic_x3');
  div2k_x4_builder = tfds.builder('div2k/bicubic_x4');
  div2k_x2_builder.download_and_prepare();
  div2k_x3_builder.download_and_prepare();
  div2k_x4_builder.download_and_prepare();

def parse_sample(features):
  hr, lr = features['hr'], features['lr'];
  lr = tf.cast(lr, dtype = tf.float32) - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
  hr = tf.cast(hr, dtype = tf.float32) - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
  return lr, hr;

def load_datasets():
  trainset_x2 = tfds.load(name = 'div2k/bicubic_x2', split = 'train', download = False);
  testset_x2 = tfds.load(name = 'div2k/bicubic_x2', split = 'validation', download = False);
  trainset_x3 = tfds.load(name = 'div2k/bicubic_x3', split = 'train', download = False);
  testset_x3 = tfds.load(name = 'div2k/bicubic_x3', split = 'validation', download = False);
  trainset_x4 = tfds.load(name = 'div2k/bicubic_x4', split = 'train', download = False);
  testset_x4 = tfds.load(name = 'div2k/bicubic_x4', split = 'validation', download = False);
  trainset_x2 = trainset_x2.map(parse_sample);
  testset_x2 = testset_x2.map(parse_sample);
  trainset_x3 = trainset_x3.map(parse_sample);
  testset_x3 = testset_x3.map(parse_sample);
  trainset_x4 = trainset_x4.map(parse_sample);
  testset_x4 = testset_x4.map(parse_sample);
  return (trainset_x2,testset_x2), (trainset_x3, testset_x3), (trainset_x4, testset_x4);

def main(unused_argv):
  if FLAGS.test == True:
    import numpy as np;
    import cv2;
    (train_x2, test_x2), (train_x3, test_x3), (train_x4, test_x4) = load_datasets();
    for lr, hr in train_x2:
      lr = lr + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
      hr = hr + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
      lr = lr.numpy().astype(np.uint8);
      hr = hr.numpy().astype(np.uint8);
      cv2.imshow('lr', lr);
      cv2.imshow('hr', hr);
      cv2.waitKey();
  else:
    download();

if __name__ == "__main__":
  add_options();
  app.run(main);

