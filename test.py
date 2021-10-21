#!/usr/bin/python3

from os.path import join;
from absl import flags, app;
import numpy as np;
import cv2;
import tensorflow as tf;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_string('model', default = join('models', 'model_x2.h5'), help = 'model path');
  flags.DEFINE_string('image', default = None, help = 'image path');

def main(unused_argv):
  img = cv2.imread(FLAGS.image);
  if img is None:
    raise Exception('invalid image');
  inputs = tf.expand_dims(img, axis = 0); # inputs.shape = (1, h, w, 3)
  inputs = tf.cast(inputs, dtype = tf.float32) - tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
  model = tf.keras.models.load_model(FLAGS.model);
  outputs = model(inputs);
  outputs = outputs + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3));
  large = tf.squeeze(outputs, axis = 0); # large.shape = (h, w, 3)
  large = large.numpy().astype(np.uint8);
  cv2.imshow('sr', large);
  cv2.waitKey();

if __name__ == "__main__":
  add_options();
  app.run(main);
