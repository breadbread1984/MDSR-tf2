#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import app, flags;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_dataset import load_datasets;
from models import *;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_enum('model', default = 'MDSR', enum_values = ['EDSR', 'MDSR'], help = 'model to train');
  flags.DEFINE_integer('batch_size', default = 16, help = 'batch size');
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'path to checkpoint directory');
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate');
  flags.DEFINE_bool('save_model', default = False, help = 'whether to save model');
  flags.DEFINE_integer('eval_steps', default = 100, help = 'how many iterations for each evaluation');
  flags.DEFINE_integer('checkpoint_steps', default = 100, help = 'how many iterations for each checkpoint');
  flags.DEFINE_enum('scale', default = '2', enum_values = ['2','3','4'], help = 'train EDSR on which scale of DIV2K');
  flags.DEFINE_integer('lr_size', default = 192, help = 'input size for low resolution image');
  flags.DEFINE_enum('method', default = 'bicubic', enum_values = ['area', 'bicubic', 'bilinear', 'gaussian', 'lanczos3', 'lanczos5', 'mitchellcubic', 'nearest'], help = 'downsample method for preprocess');

def main(unused_argv):
  # 1) create dataset
  (trainset_x2,testset_x2), (trainset_x3, testset_x3), (trainset_x4, testset_x4) = load_datasets((FLAGS.lr_size, FLAGS.lr_size), FLAGS.method);
  trainset_x2 = iter(trainset_x2.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  testset_x2 = iter(testset_x2.batch(1).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  trainset_x3 = iter(trainset_x3.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  testset_x3 = iter(testset_x3.batch(1).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  trainset_x4 = iter(trainset_x4.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  testset_x4 = iter(testset_x4.batch(1).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  trainsets = [trainset_x2, trainset_x3, trainset_x4];
  testsets = [testset_x2, testset_x3, testset_x4];
  # 2) create model
  if FLAGS.model == 'MDSR':
    model_x2, model_x3, model_x4 = MDSR(scales = [2,3,4]);
    models = [model_x2, model_x3, model_x4];
  elif FLAGS.model == 'EDSR':
    model = EDSR();
  else:
    raise Exception('unknown model');
  # 3) optimizer
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr);
  # 4) restore from existing checkpoint
  if not exists(FLAGS.checkpoint): mkdir(FLAGS.checkpoint);
  if FLAGS.model == 'MDSR':
    checkpoint = tf.train.Checkpoint(model_x2 = model_x2, model_x3 = model_x3, model_x4 = model_x4, optimizer = optimizer);
  elif FLAGS.model == 'EDSR':
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
  else:
    raise Exception('unknown model');
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.checkpoint, 'ckpt')));
  if FLAGS.save_model:
    if not exists('models'): mkdir('models');
    if FLAGS.model == 'MDSR':
      model_x2.save(join('models', 'model_x2.h5'));
      model_x3.save(join('models', 'model_x3.h5'));
      model_x4.save(join('models', 'model_x4.h5'));
    elif FLAGS.model == 'EDSR':
      model.save(join('models', 'model.h5'));
    exit();
  # 5) log
  log = tf.summary.create_file_writer('checkpoints');
  if FLAGS.model == 'MDSR':
    x2_loss = tf.keras.metrics.Mean(name = 'x2_loss', dtype = tf.float32);
    x3_loss = tf.keras.metrics.Mean(name = 'x3_loss', dtype = tf.float32);
    x4_loss = tf.keras.metrics.Mean(name = 'x4_loss', dtype = tf.float32);
    avg_losses = [x2_loss, x3_loss, x4_loss];
  elif FLAGS.model == 'EDSR':
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  else:
    raise Exception('unknown model');
  while True:
    if FLAGS.model == 'MDSR':
      idx = optimizer.iterations % 3;
      model = models[idx];
      avg_loss = avg_losses[idx];
    elif FLAGS.model == 'EDSR':
      idx = {'2': 0, '3': 1, '4': 2}[FLAGS.scale];
    else:
      raise Exception('unknown model');
    trainset = trainsets[idx];
    testset = testsets[idx];
    lr, hr = next(trainset);
    with tf.GradientTape() as tape:
      preds = model(lr);
      loss = tf.keras.losses.MeanAbsoluteError()(hr, preds);
    grads = tape.gradient(loss, model.trainable_variables);
    optimizer.apply_gradients(zip(grads, model.trainable_variables));
    avg_loss.update_state(loss);
    if optimizer.iterations % FLAGS.checkpoint_steps == 1:
      checkpoint.save(join(FLAGS.checkpoint, 'ckpt'));
    if optimizer.iterations % FLAGS.eval_steps == 1:
      lr, hr = next(testset);
      preds = model(lr);
      image = tf.cast(preds + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), dtype = tf.uint8);
      hr = tf.cast(hr + tf.reshape([114.444 , 111.4605, 103.02  ], (1,1,1,3)), dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('x' + str([2,3,4][idx]) + '_loss', avg_loss.result(), step = optimizer.iterations);
        tf.summary.image('predicted', image, step = optimizer.iterations);
        tf.summary.image('ground truth', hr, step = optimizer.iterations);
      print('#%d loss: %f' % (optimizer.iterations, avg_loss.result()));
      avg_loss.reset_states();

if __name__ == "__main__":
  add_options();
  app.run(main);

