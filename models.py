#!/usr/bin/python3

from math import log2;
import tensorflow as tf;

def Preprocess(n_feats = 64, n_resblocks = 2, res_scale = 1, **kwargs):
  inputs = tf.keras.Input((None, None, n_feats));
  results = inputs;
  for i in range(n_resblocks):
    skip = results;
    results = tf.keras.layers.Conv2D(n_feats, kernel_size = (5,5), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(n_feats, kernel_size = (5,5), padding = 'same')(results);
    results = tf.keras.layers.Lambda(lambda x, s: x * s, arguments = {'s': res_scale})(results);
    results = tf.keras.layers.Add()([results, skip]);
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def Body(n_feats = 64, n_resblocks = 16, res_scale = 1, **kwargs):
  inputs = tf.keras.Input((None, None, n_feats));
  results = inputs;
  global_skip = results;
  for i in range(n_resblocks):
    skip = results;
    results = tf.keras.layers.Conv2D(n_feats, kernel_size = (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
    results = tf.keras.layers.Conv2D(n_feats, kernel_size = (3,3), padding = 'same')(results);
    results = tf.keras.layers.Lambda(lambda x, s: x * s, arguments = {'s': res_scale})(results);
    results = tf.keras.layers.Add()([results, skip]);
  results = tf.keras.layers.Conv2D(n_feats, kernel_size = (3,3), padding = 'same')(results);
  results = tf.keras.layers.Add()([results, global_skip]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def UpSample(n_feats = 64, scale = 2, **kwargs):
  inputs = tf.keras.Input((None, None, n_feats));
  results = inputs;
  if log2(scale) == round(log2(scale), 0):
    for i in range(int(log2(scale))):
      results = tf.keras.layers.Conv2D(4 * n_feats, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch, h, w, 4 * c)
      results = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(results); # results.shape = (batch, h * 2, w * 2, c)
  elif scale == 3:
    results = tf.keras.layers.Conv2D(9 * n_feats, kernel_size = (3,3), padding = 'same')(results); # results.shape = (batch, h, w, 9 * c)
    results = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 3))(results); # results.shape = (batch, h * 3, w * 3, c)
  else:
    raise Exception('unimplemented scale');
  return tf.keras.Model(inputs = inputs, outputs = results, **kwargs);

def EDSR(n_feats = 64, n_resblocks = 16, res_scale = 1, scale = 2):
  inputs = tf.keras.Input((None, None, 3));
  # head
  results = tf.keras.layers.Conv2D(n_feats, kernel_size = (3,3), padding = 'same', name = 'head')(inputs);
  # body
  results = Body(n_feats, n_resblocks, res_scale, name = 'body')(results);
  # upsample
  results = UpSample(n_feats, scale, name = 'upsample')(results);
  # tail
  results = tf.keras.layers.Conv2D(3, kernel_size = (3,3), padding = 'same', name = 'tail')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def MDSR(n_feats = 64, n_resblocks = 16, res_scale = 1, scales = [2,3,4]):
  # NOTE: the parts are all defined as functional to facilitate weights restoring
  head = tf.keras.layers.Conv2D(n_feats, kernel_size = (3,3), padding = 'same', name = 'head');
  preprocesses = [Preprocess(n_feats, n_resblocks, res_scale, name = 'preprocess' + str(idx)) for idx, _ in enumerate(scales)];
  body = Body(n_feats, n_resblocks, res_scale, name = 'body');
  upsamples = [UpSample(n_feats, s, name = 'upsample' + str(idx)) for idx, s in enumerate(scales)];
  tail = tf.keras.layers.Conv2D(3, kernel_size = (3,3), padding = 'same', name = 'tail');
  
  models = list();
  for idx, _ in enumerate(scales):
    inputs = tf.keras.Input((None, None, 3));
    # head
    results = head(inputs);
    # preprocess
    results = preprocesses[idx](results);
    # body
    results = body(results);
    # upsample
    results = upsamples[idx](results);
    # tail
    results = tail(results);
    models.append(tf.keras.Model(inputs = inputs, outputs = results));
  # NOTE: the returned models share weights at head, body and tail
  return models;

if __name__ == "__main__":
  edsr = EDSR(scale = 3);
  import numpy as np;
  inputs = np.ones((1,10,10,3)) * 255;
  outputs = edsr(inputs);
  print(outputs.shape);
  edsr.save('edsr.h5');
  models = MDSR();
  for i, model in enumerate(models):
    outputs = model(inputs);
    print(outputs.shape);
    model.save('model' + str(i) + '.h5');
