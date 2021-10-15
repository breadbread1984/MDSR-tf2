#!/usr/bin/python3

from absl import app, flags;
import tensorflow as tf;
import tensorflow_datasets as tfds;

def add_options():
  flags.DEFINE_bool('test', False, help = 'test dataset input pipeline');

def download():
  div2k_x2_builder = tfds.build('div2k/bicubic_x2');
  div2k_x3_builder = tfds.build('div2k/bicubic_x3');
  div2k_x4_builder = tfds.build('div2k/bicubic_x4');
  div2k_x2_builder.download_and_prepare();
  div2k_x3_builder.download_and_prepare();
  div2k_x4_builder.download_and_prepare();

def parse_sample(features):
  hr, lr = features['hr'], features['lr'];
  lr = lr - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
  hr = hr - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,3));
  return lr, hr;

def load_datasets():
  trainset_x2 = tfds.load(name = 'div2k/bicubic_x2', split = 'train', download = False);
  testset_x2 = tfds.load(name = 'div2k/bicubic_x2', split = 'validation', download = False);
  trainset_x3 = tfds.load(name = 'div2k/bicubic_x3', split = 'train', download = False);
  testset_x3 = tfds.load(name = 'div2k/bicubic_x3', split = 'validation', download = False);
  trainset_x4 = tfds.load(name = 'div2k/bicubic_x4', split = 'train', download = False);
  testset_x4 = tfds.load(name = 'div2k/bicubic_x4', split = 'validation', download = False);

if __name__ == "__main__":
  add_options();
  if FLAGS.test:
    pass;
  else:
    download();

