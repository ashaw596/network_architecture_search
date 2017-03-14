# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

import cifar10_input
import cifar10_cached
from tqdm import tqdm
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
UPSAMPLED_IMAGE_FILE = "cifar10_data/img_order.npy"
UPSAMPLED_IMAGE_LABELS = "cifar10_data/lab_order.npy"


TEST_IMAGE_FILE = "cifar10_data/img_test.npy"
TEST_IMAGE_LABELS = "cifar10_data/lab_test.npy"

BATCH_SIZE = 100
IMAGE_SIZE = cifar10_cached.IMAGE_SIZE

class CifarBatchGenerator(object):
  def __init__(self, upsampled_images, upsampled_labels):
    self.upsampled_images = upsampled_images
    self.upsampled_labels = upsampled_labels

    self.i = 0
    self.length = self.upsampled_images.shape[0]
    self.current_cropped = None
    self.current_labels = None

    self.crop_batch()

  def crop_batch(self):
    self.current_cropped, self.current_labels = manual_crop(self.upsampled_images, self.upsampled_labels)
    self.i = 0

  def next_batch(self, size):
    if self.i + size > self.length:
      assert(False)
    cropped_copy = self.current_cropped[self.i:self.i+size, :, :, :]
    labels = self.current_labels[self.i:self.i+size, :]
    self.i += size
    if self.i == self.length:
      self.crop_batch()
    return cropped_copy, labels

def get_batch_generator(upsampled_images, upsampled_labels):
  return CifarBatchGenerator(upsampled_images, upsampled_labels)

def unison_shuffled_copies(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

def manual_crop(upsampled_images, upsampled_labels):
  shuffled_images, shuffled_labels = unison_shuffled_copies(upsampled_images, upsampled_labels)
  length = upsampled_images.shape[0]
  image_size = cifar10_cached.IMAGE_SIZE
  diff = cifar10_cached.UPSAMPLE_SIZE - cifar10_cached.IMAGE_SIZE
  start_indexes = np.random.randint(diff + 1, size=[length, 2])
  to_flip = np.random.randint(2, size=[length])

  cropped = np.zeros([length, image_size, image_size, 3],dtype=np.float32)
  for i in range(length):
    x = start_indexes[i,0]
    y = start_indexes[i,0]
    cropped[i,:,:,:] = shuffled_images[i,x:x+image_size, y:y+image_size, :]
    if to_flip[i]==0:
      cropped[i,:,:,:] = cropped[i,:,::-1,:]
      #print("flip")
    #print(i)
  

  cropped.setflags(write=False)
  shuffled_labels.setflags(write=False)
  return cropped, shuffled_labels

def load_upsampled_files(image_file=UPSAMPLED_IMAGE_FILE, labels_file=UPSAMPLED_IMAGE_LABELS):
  upsampled_images = np.load(image_file)
  upsampled_labels = np.load(labels_file)
  upsampled_images.setflags(write=False)
  upsampled_labels.setflags(write=False)
  return upsampled_images, upsampled_labels

def load_test_files(image_file=TEST_IMAGE_FILE, labels_file=TEST_IMAGE_LABELS):
  test_images = np.load(image_file)
  test_labels = np.load(labels_file)
  test_images.setflags(write=False)
  test_labels.setflags(write=False)
  return test_images, test_labels

def main():
  pass

def test_load_upsampled():
  upsampled_images, upsampled_labels = load_upsampled_files()
  cropped_images, labels = manual_crop(upsampled_images, upsampled_labels)
  x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
  y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10))
  #t= tf.expand_dims(tf.expand_dims(tf.expand_dims(y, 1),2),3)
  #z = tf.multiply(x,tf.to_float(t))
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # avoid using all vram for GTX 970
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))#config=tf.ConfigProto()
  sess.run([
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
  ])
  for i in tqdm(range(500)):
    index= i*BATCH_SIZE
    x_, y_ = sess.run([x, y], feed_dict={x:np.copy(cropped_images[index:index+BATCH_SIZE, :,:,:]), y:labels[index:index+BATCH_SIZE, :]})
  return x_, y_
  #return cropped_images, labels
def main_queue():
  cropped_images, labels = crop_upsampled(UPSAMPLED_IMAGE_FILE, UPSAMPLED_IMAGE_LABELS, 50)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # avoid using all vram for GTX 970
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))#config=tf.ConfigProto()
  #sess = tf.Session()
  coord = tf.train.Coordinator()
  sess.run([
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
  ])
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  all_images = np.zeros([0,32,32,3])
  all_labels = np.zeros([0])
  try:
    for i in tqdm(range(2000)):
      raw_images, raw_labels = sess.run([cropped_images, labels])
      #print(raw_images.shape)
      #all_images = np.concatenate([all_images,raw_images])
      #all_labels = np.concatenate([all_labels,raw_labels])
  finally:
    coord.request_stop()
    coord.join(threads)

  return all_images, all_labels

def load_test_main():
  #maybe_download_and_extract()
  sess = tf.Session()
  coord = tf.train.Coordinator()
  images, labels = normalize(True, 10000)
  sess.run([
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
  ])
  #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  #coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  all_images = np.zeros([0,32,32,3], dtype=np.float32)
  all_labels = np.zeros([0], dtype=np.uint8)
  try:
      # in most cases coord.should_stop() will return True
      # when there are no more samples to read
      # if num_epochs=0 then it will run for ever
    while not coord.should_stop():
          # will start reading, working data from input queue
          # and "fetch" the results of the computation graph
          # into raw_images and raw_labels
      raw_images, raw_labels = sess.run([images, labels])
      print(raw_images.shape)
      all_images = np.concatenate([all_images,raw_images])
      all_labels = np.concatenate([all_labels,raw_labels])
  except Exception:
    pass
  finally:
    coord.request_stop()
    coord.join(threads)
  #img2, lab2 = sess.run([images, labels])
  #img3, lab3 = sess.run([images, labels])
  #img4, lab4 = sess.run([images, labels])
  #img5, lab5 = sess.run([images, labels])
  #images = np.concatenate([img1,img2,img3,img4,img5])
  #labels = np.concatenate([lab1,lab2,lab3,lab4,lab5])
  #return images, labels
  return all_images, all_labels

def upsample_main():
  #maybe_download_and_extract()
  sess = tf.Session()
  coord = tf.train.Coordinator()
  images, labels = upsample(False, 10000)
  sess.run([
      tf.local_variables_initializer(),
      tf.global_variables_initializer(),
  ])
  #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  #coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  all_images = np.zeros([0,40,40,3], dtype=np.float32)
  all_labels = np.zeros([0], dtype=np.uint8)
  try:
      # in most cases coord.should_stop() will return True
      # when there are no more samples to read
      # if num_epochs=0 then it will run for ever
    while not coord.should_stop():
          # will start reading, working data from input queue
          # and "fetch" the results of the computation graph
          # into raw_images and raw_labels
      raw_images, raw_labels = sess.run([images, labels])
      print(raw_images.shape)
      all_images = np.concatenate([all_images,raw_images])
      all_labels = np.concatenate([all_labels,raw_labels])
  except Exception:
    pass
  finally:
    coord.request_stop()
    coord.join(threads)
  #img2, lab2 = sess.run([images, labels])
  #img3, lab3 = sess.run([images, labels])
  #img4, lab4 = sess.run([images, labels])
  #img5, lab5 = sess.run([images, labels])
  #images = np.concatenate([img1,img2,img3,img4,img5])
  #labels = np.concatenate([lab1,lab2,lab3,lab4,lab5])
  #return images, labels
  return all_images, all_labels

def crop_upsampled(image_file, labels_file, batch_size):
  upsampled_images = np.load(image_file)
  upsampled_labels = np.load(labels_file)
  print("loaded")
  cropped_images, labels = cifar10_cached.distorted_inputs(upsampled_images, upsampled_labels, batch_size)
  return cropped_images, labels

def up(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.upsample_in(eval_data=eval_data,
                                        data_dir=data_dir)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def upsample(eval_data, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.upsample_inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def normalize(eval_data, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.normalize_inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

if __name__ == "__main__":
    main()