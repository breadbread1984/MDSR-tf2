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

mean = [114.444 , 111.4605, 103.02  ]

if __name__ == "__main__":
  add_options();
  if FLAGS.test:
    pass;
  else:
    download();

