from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import math
import tensorflow.contrib.slim as slim

def main():
    #batch_size = 100
    num_classes = 10
    
    #[batch_size, time_series, 4]
    batchX_placeholder = tf.placeholder(tf.float32, [None, None, 4])
    state_size = 35
    num_layers = 2
    lstm_layer = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    multiRNN = tf.nn.rnn_cell.MultiRNNCell([lstm_layer]*num_layers, state_is_tuple=True)
    outputs, final_state = tf.nn.dynamic_rnn(multiRNN, batchX_placeholder, initial_state=None, dtype=tf.float32)

    W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
    
    #Squash batch for multiplication
    states_series = tf.reshape(outputs, [-1, state_size])

    logits = tf.matmul(states_series, W2) + b2 #Broadcasted addition

    return outputs, final_state, batchX_placeholder

if __name__ == "__main__":
    main()