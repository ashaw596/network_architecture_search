from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import math
import tensorflow.contrib.slim as slim

class QNetworkLSTM(object):
    def __init__(self, args, num_actions, scope_name = "global"):
        self.discount_factor = 1.0
        self.target_update_frequency = 1000
        self.gradient_clip = 10
        self.learning_rate = 0.0006
        self.num_actions = num_actions
        self.num_layers = 2
        self.layer_size = 35
        self.total_updates = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)  # avoid using all vram for GTX 970
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))#, log_device_placement=True))#config=tf.ConfigProto()

            with tf.variable_scope(scope_name) as scope:
                self.batchX_placeholder = tf.placeholder(tf.float32, [None, None, self.num_actions], name="batchX_placeholder")
                self.rewards_placeholder = tf.placeholder(tf.float32, [None, None], name="rewards_placeholder")
                self.terminal_placeholder = tf.placeholder(tf.float32, [None, None], name="terminal_placeholder")

                
                #stupid way to pad with 1/num_classes
                padding = [[0,0], [1,0], [0,0]]
                batchX = self.batchX_placeholder - 1.0/self.num_actions
                batchX = tf.pad(batchX, padding, "CONSTANT")
                batchX = batchX + 1.0/self.num_actions

                shape = tf.shape(batchX)
                batch_size, batchX_time_length, classes = shape[0], shape[1], shape[2]
                
                lstm_layer = tf.nn.rnn_cell.LSTMCell(self.layer_size, state_is_tuple=True)
                multiRNN = tf.nn.rnn_cell.MultiRNNCell([lstm_layer]*self.num_layers, state_is_tuple=True)
                with tf.variable_scope("policy"):
                    '''
                    #training starting_state
                    zero_state = multiRNN.zero_state(1, tf.float32)
                    layer1_start = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(tf.Variable(zero_state[0][0], trainable=True), [batch_size, 1], name='layer1_c'), tf.tile(tf.Variable(zero_state[0][1], trainable=True), [batch_size, 1]))
                    layer2_start = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(tf.Variable(zero_state[1][0], trainable=True), [batch_size, 1]), tf.tile(tf.Variable(zero_state[1][1], trainable=True), [batch_size, 1]))
                    policy_outputs, policy_final_state = tf.nn.dynamic_rnn(multiRNN, batchX, initial_state=(layer1_start, layer2_start), dtype=tf.float32)
                    print(policy_initial_state)
                    
                    '''


                    policy_outputs, policy_final_state = tf.nn.dynamic_rnn(multiRNN, batchX, initial_state=None, dtype=tf.float32)

                    #Slice off the last step which isn't trained
                    last_output = tf.slice(policy_outputs, [0, batchX_time_length-1, 0], [-1, -1, -1])

                    policy_outputs = tf.slice(policy_outputs, [0, 0, 0], [-1, batchX_time_length-1, -1])
                    
                    policy_W2 = tf.Variable(np.random.rand(self.layer_size, self.num_actions), dtype=tf.float32, name="W2")
                    policy_b2 = tf.Variable(np.zeros((1,self.num_actions)), dtype=tf.float32, name="b2")
                    #Squash batch for multiplication
                    policy_states_series = tf.reshape(policy_outputs, [-1, self.layer_size])
                    policy_logits = tf.matmul(policy_states_series, policy_W2) + policy_b2 #Broadcasted addition]


                    self.last_policy_q = tf.matmul(tf.reshape(last_output, [-1, self.layer_size]), policy_W2) + policy_b2

                with tf.variable_scope("target"):
                    target_outputs, target_final_state = tf.nn.dynamic_rnn(multiRNN, batchX, initial_state=None, dtype=tf.float32)

                    # change this n+1
                    target_outputs = tf.slice(target_outputs, [0, 1, 0], [-1, -1, -1])

                    target_W2 = tf.Variable(np.random.rand(self.layer_size, self.num_actions),dtype=tf.float32, name="W2")
                    target_b2 = tf.Variable(np.zeros((1,self.num_actions)), dtype=tf.float32, name="b2")
                    #Squash batch for multiplication
                    target_states_series = tf.reshape(target_outputs, [-1, self.layer_size])
                    target_logits = tf.matmul(target_states_series, target_W2) + target_b2 #Broadcasted addition

                with tf.variable_scope("loss"):
                    self.max_action_values = tf.reduce_max(target_logits, axis=1)
                    rewards = tf.reshape(self.rewards_placeholder, [-1])
                    terminals = tf.reshape(self.terminal_placeholder, [-1])
                    self.targets = tf.stop_gradient(rewards + (self.discount_factor * self.max_action_values * (1 - tf.to_float(terminals))))
                    actions = tf.reshape(self.batchX_placeholder, [-1, self.num_actions])
                    self.predictions = tf.reduce_sum(tf.mul(policy_logits, actions), 1)
                    difference = tf.abs(self.predictions - self.targets)
                    self.diff_error = (0.5 * tf.square(difference))
                    self.loss = tf.reduce_sum(self.diff_error)


                self.policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name + "/policy")
                self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name + "/target")
                self.update_target_variables_op = []
                for policy_var,target_var in zip(self.policy_vars, self.target_vars):
                    self.update_target_variables_op.append(target_var.assign(policy_var))
                    print (policy_var.name, "=>", target_var.name)


                #Optimization
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

                #Gradient clipping
                grads_and_vars = self.optimizer.compute_gradients(self.loss, self.policy_vars)
                grads = [gv[0] for gv in grads_and_vars]
                params = [gv[1] for gv in grads_and_vars]

                if self.gradient_clip > 0:
                    grads = tf.clip_by_global_norm(grads, self.gradient_clip)[0]
                self.train_op = self.optimizer.apply_gradients(zip(grads, params))

                #Initialize Network
                self.sess.run([
                    tf.local_variables_initializer(),
                    tf.global_variables_initializer(),
                ])
                print("Network Initialized")


    def train(self, actions, rewards, is_terminals):
        ''' train network on batch of experiences

        Args:
            o1: first observations
            a: actions taken
            r: rewards received
            o2: succeeding observations
        '''
        train_op, loss = self.sess.run([self.train_op, self.loss], 
            feed_dict={self.batchX_placeholder:actions, self.rewards_placeholder:rewards, self.terminal_placeholder:is_terminals})
        self.total_updates += 1
        if self.total_updates % self.target_update_frequency == 0:
            self.sess.run(self.update_target_variables_op)
            print("update")

        return loss

    def inference(self, actions):
        ''' Get state-action value predictions for an observation 

        Args:
            observation: the observation
        '''
        #print np.squeeze(self.sess.run(self.target_q_layer, feed_dict={self.observation:obs}))
        #obs = obs.append(obs[0], axis = 0)
        last_q = self.sess.run(self.last_policy_q, 
            feed_dict={self.batchX_placeholder:actions})
        return last_q


def main():
    network = QNetworkLSTM(args=None, num_actions=4, scope_name="global")
    
    q = network.inference(np.zeros([1,0,4]))
    print(q)
    #terminals = [[0,1],[0,1],[0,1]]
    #rewards = [[10,-6],[10,4], [2,1]]
    #actions = [[[1, 0, 0, 0],[0, 0, 0, 1]],[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 1, 0],[0, 0, 0, 1]]]
    actions = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    rewards = [[0.7, 0.01, -0.01, 0.1]]
    terminals = [[0,0,0,1]]

    feed_dict = {network.terminal_placeholder:terminals, network.rewards_placeholder:rewards, network.batchX_placeholder:actions}


    for i in range(2010):
        #if i%100==0:
        #    network.sess.run(network.update_target_variables_op)
        #    print("update")
        loss = network.train(actions=actions, rewards=rewards, is_terminals=terminals)
        pred, targets, last_q, max_action = network.sess.run([network.predictions, network.targets, network.last_policy_q, network.max_action_values], feed_dict=feed_dict)
        print(i)
        print(max_action)
        print(last_q)
        print(loss)
        print(pred)
        print(targets)


    q_values = network.inference(actions)
    print(q_values)
    '''
    o, f, l, tl, batch, batchX_time_length, max_action_values, predictions, difference, lo, targets, rewards = sess.run(
        [policy_outputs, policy_final_state, policy_logits, target_logits, batchX, batchX_time_length, max_action_values, predictions, difference, loss, targets, rewards], 
        feed_dict=feed_dict)
    
    #print(batchX_time_length)
    print(predictions)
    print(targets)
    print(max_action_values)
    print(rewards)
    #print(l)
    #print(tl)
    #print(batch)

    print("diff")
    print(difference)
    print(lo)
    
    print(o)
    print(lo)

    #print(o)
    #print(f)
    #return o, f, l, batch
    #return outputs, final_state, self.batchX_placeholder, logits
    '''
if __name__ == "__main__":
    main()