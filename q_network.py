import tensorflow as tf
import os
import numpy as np
import math
import tensorflow.contrib.slim as slim


class QNetwork():

	def __init__(self, args, num_actions, scope_name = "global"):
		with tf.variable_scope(scope_name) as scope:
			''' Build tensorflow graph for deep q network '''

			print("Initializing Q-Network")

			self.discount_factor = args.discount_factor
			self.target_update_frequency = args.target_update_frequency
			self.total_updates = 0
			self.path = '../saved_models/' + args.game + '/' + args.agent_type + '/' + args.agent_name
			if not os.path.exists(self.path):
				os.makedirs(self.path)
			self.name = args.agent_name
			self.priority_replay = args.priority_replay
			self.enable_constraints = args.enable_constraints
			self.alpha = args.alpha
			self.skip = args.skip

			# input placeholders
			self.observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="observation")
			self.actions = tf.placeholder(tf.float32, shape=[None, num_actions], name="actions") # one-hot matrix because tf.gather() doesn't support multidimensional indexing yet
			self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
			self.next_observation = tf.placeholder(tf.float32, shape=[None, args.screen_dims[0], args.screen_dims[1], args.history_length], name="next_observation")
			self.terminals = tf.placeholder(tf.float32, shape=[None], name="terminals")
			self.normalized_observation = self.observation / 255.0
			self.normalized_next_observation = self.next_observation / 255.0
			self.importance_sampling_weights = tf.placeholder(tf.float32, shape=[None], name="importance_sampling_weights")


			self.max_ls = tf.placeholder(tf.float32, shape=[None], name="max_ls")
			self.min_us = tf.placeholder(tf.float32, shape=[None], name="min_us")
			#self.real_discounted_reward = tf.placeholder(tf.float32, shape=[None], name="real_discounted_reward")
			#self.min_real_discounted_reward = tf.placeholder(tf.float32, shape=[None], name="min_real_discounted_reward")
			#self.max_real_discounted_reward = tf.placeholder(tf.float32, shape=[None], name="max_real_discounted_reward")

			num_conv_layers = len(args.conv_kernel_shapes)
			assert(num_conv_layers == len(args.conv_strides))
			num_dense_layers = len(args.dense_layer_shapes)

			last_policy_layer = None
			last_target_layer = None
			self.update_target = []
			self.policy_network_params = []
			self.param_names = []

			# initialize convolutional layers
			for layer in range(num_conv_layers):
				policy_input = None
				target_input = None
				if layer == 0:
					if args.double_dqn:
						policy_input = tf.concat(values=[self.normalized_observation, self.normalized_next_observation], concat_dim=0, name='concat')
					else:
						policy_input = self.normalized_observation
					target_input = self.normalized_next_observation
				else:
					policy_input = last_policy_layer
					target_input = last_target_layer
				#last_layers = self.conv_relu(policy_input, target_input, 
				#	args.conv_kernel_shapes[layer], args.conv_strides[layer], layer)
				#last_policy_layer = last_layers[0]
				#last_target_layer = last_layers[1]
				
				last_policy_layer = slim.conv2d(activation_fn=tf.nn.relu,
					inputs=policy_input,num_outputs=args.conv_kernel_shapes[layer][-1],
					kernel_size=args.conv_kernel_shapes[layer][:-2],
					stride=args.conv_strides[layer][1:3],
					padding='VALID',
					trainable=True,
					scope="policy/conv" + str(layer))
				last_target_layer = slim.conv2d(activation_fn=tf.nn.relu,
					inputs=target_input,num_outputs=args.conv_kernel_shapes[layer][-1],
					kernel_size=args.conv_kernel_shapes[layer][:-2],
					stride=args.conv_strides[layer][1:3],
					padding='VALID',
					trainable=False,
					scope="target/conv" + str(layer))

			# initialize fully-connected layers
			for layer in range(num_dense_layers):
				policy_input = None
				target_input = None
				if layer == 0:
					input_size = args.dense_layer_shapes[0][0]
					policy_input = tf.reshape(last_policy_layer, shape=[-1, input_size])
					target_input = tf.reshape(last_target_layer, shape=[-1, input_size])
				else:
					policy_input = last_policy_layer
					target_input = last_target_layer
				last_policy_layer = slim.fully_connected(
					policy_input, 
					args.dense_layer_shapes[layer][1],
					activation_fn=tf.nn.relu,
					trainable=True,
					scope="policy/fc" + str(layer))
				last_target_layer = slim.fully_connected(
					target_input, 
					args.dense_layer_shapes[layer][1],
					activation_fn=tf.nn.relu,
					trainable=False,
					scope="target/fc" + str(layer))

			# initialize q_layer
			last_q_layer = slim.fully_connected(
				last_policy_layer, 
				num_actions,
				activation_fn=None,
				trainable=True,
				scope="policy/fc" + str(num_dense_layers))
			if args.double_dqn:
				obs_length = tf.shape(self.normalized_observation, out_type=tf.int32)[0]
				self.policy_q_layer = tf.slice(input_=last_q_layer, begin=[0,0] , size=[obs_length,-1], name=None)
				self.next_policy_q_layer = tf.slice(input_=last_q_layer, begin=[obs_length,0] , size=[-1,-1], name=None)
			else:
				self.policy_q_layer = last_q_layer

			self.target_q_layer = slim.fully_connected(
				last_target_layer, 
				num_actions,
				activation_fn=None,
				trainable=False,
				scope="target/fc" + str(num_dense_layers))

			policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name +"/policy")
			target_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope.name + "/target")
			for policy_var,target_var in zip(policy_vars,target_vars):
				self.update_target.append(target_var.assign(policy_var))
				print (policy_var.name, target_var.name)
			self.policy_network_params = policy_vars

			

			losses = self.build_loss(args.error_clipping, num_actions, args.double_dqn)
			if self.priority_replay:
				self.losses = self.importance_sampling_weights*losses
			else:
				self.losses = losses

			self.loss = tf.reduce_sum(self.losses)

			if (args.optimizer == 'rmsprop') and (args.gradient_clip <= 0):
				self.train_op = tf.train.RMSPropOptimizer(
					args.learning_rate, decay=args.rmsprop_decay, momentum=0.0, epsilon=args.rmsprop_epsilon).minimize(self.loss)
			elif (args.optimizer == 'graves_rmsprop') or (args.optimizer == 'rmsprop' and args.gradient_clip > 0):
				self.train_op = self.build_rmsprop_optimizer(args.learning_rate, args.rmsprop_decay, args.rmsprop_epsilon, args.gradient_clip, args.optimizer)

			self.saver = tf.train.Saver(policy_vars)

			if not args.watch:
				param_hists = [tf.histogram_summary(param.name, param) for param in self.policy_network_params]
				self.param_summaries = tf.merge_summary(param_hists)

			# start tf session
			gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666666)  # avoid using all vram for GTX 970
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

			if args.watch:
				print("Loading Saved Network...")
				load_path = tf.train.latest_checkpoint(self.path)
				self.saver.restore(self.sess, load_path)
				print("Network Loaded")		
			else:
				self.sess.run(tf.initialize_all_variables())
				print("Network Initialized")
				self.summary_writer = tf.train.SummaryWriter('../records/' + args.game + '/' + args.agent_type + '/' + args.agent_name + '/params', self.sess.graph)

	def inference(self, obs):
		''' Get state-action value predictions for an observation 

		Args:
			observation: the observation
		'''
		#print np.squeeze(self.sess.run(self.target_q_layer, feed_dict={self.observation:obs}))
		#obs = obs.append(obs[0], axis = 0)
		next_obs = np.empty([0, 84, 84, 4])
		return np.squeeze(self.sess.run(self.policy_q_layer, feed_dict={self.observation:obs, self.next_observation:next_obs}))

	def target_inference(self, obs):
		return self.sess.run(self.target_q_layer, feed_dict={self.next_observation:obs})

	def build_loss(self, error_clip, num_actions, double_dqn):
		''' build loss graph '''
		with tf.name_scope("loss"):

			predictions = tf.reduce_sum(tf.mul(self.policy_q_layer, self.actions), 1)

			max_action_values = None
			if double_dqn: # Double Q-Learning:
				max_action_values = tf.reduce_sum(self.target_q_layer * tf.one_hot(tf.argmax(self.next_policy_q_layer, axis=1), depth=num_actions), axis=1)
				# tf.gather doesn't support multidimensional indexing yet, so we flatten output activations for indexing
				#indices = tf.range(0, tf.size(max_actions) * num_actions, num_actions) + max_actions
				#max_action_values = tf.gather(tf.reshape(self.target_q_layer, shape=[-1]), indices)
			else:
				max_action_values = tf.reduce_max(self.target_q_layer, 1)

			targets = tf.stop_gradient(self.rewards + (self.discount_factor * max_action_values * (1 - self.terminals)))

			difference = tf.abs(predictions - targets)
			#diff_error = tf.square(difference)

			penalty_coeff = 4

			maxConstraintDiff = tf.nn.relu(self.max_ls - predictions)
			minConstraintDiff = tf.nn.relu(predictions - self.min_us)
			#minConstraintError = tf.stop_gradient(penalty_coeff * tf.square(tf.nn.relu(predictions - self.min_real_discounted_reward)))
			#minConstraintError = 0

			#TODO change
			if error_clip >= 0:
				quadratic_part = tf.clip_by_value(minConstraintDiff, 0.0, error_clip)
				linear_part = minConstraintDiff - quadratic_part
				minConstraintError = penalty_coeff * (0.5 * tf.square(quadratic_part) + (error_clip * linear_part))
			else:
				minConstraintError = penalty_coeff * (0.5 * tf.square(minConstraintDiff))

			if error_clip >= 0:
				quadratic_part = tf.clip_by_value(maxConstraintDiff, 0.0, error_clip)
				linear_part = maxConstraintDiff - quadratic_part
				maxConstraintError = penalty_coeff * (0.5 * tf.square(quadratic_part) + (error_clip * linear_part))
			else:
				maxConstraintError = penalty_coeff * (0.5 * tf.square(maxConstraintDiff))

			if error_clip >= 0:
				quadratic_part = tf.clip_by_value(difference, 0.0, error_clip)
				linear_part = difference - quadratic_part
				diff_error = (0.5 * tf.square(quadratic_part)) + (error_clip * linear_part)
			else:
				diff_error = (0.5 * tf.square(difference))

			if self.enable_constraints:
				return diff_error + maxConstraintError + minConstraintError
			else:
				return diff_error

	def train(self, o1, a, r, o2, t, l, u, w):
		''' train network on batch of experiences

		Args:
			o1: first observations
			a: actions taken
			r: rewards received
			o2: succeeding observations
		'''

		train_op, losses, loss = self.sess.run([self.train_op, self.losses, self.loss], 
			feed_dict={self.observation:o1, self.actions:a, self.rewards:r, self.next_observation:o2, self.terminals:t, self.max_ls:l, self.min_us:u, self.importance_sampling_weights:w})
		self.total_updates += 1
		if self.total_updates % self.target_update_frequency == 0:
			self.sess.run(self.update_target)

		return loss, losses


	def save_model(self, epoch):

		self.saver.save(self.sess, self.path + '/' + self.name + '.ckpt', global_step=epoch)


	def build_rmsprop_optimizer(self, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip, version):

		with tf.name_scope('rmsprop'):
			optimizer = None
			if version == 'rmsprop':
				optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)
			elif version == 'graves_rmsprop':
				optimizer = tf.train.GradientDescentOptimizer(learning_rate)

			grads_and_vars = optimizer.compute_gradients(self.loss)
			grads = [gv[0] for gv in grads_and_vars]
			params = [gv[1] for gv in grads_and_vars]

			if gradient_clip > 0:
				grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

			if version == 'rmsprop':
				return optimizer.apply_gradients(zip(grads, params))
			elif version == 'graves_rmsprop':
				square_grads = [tf.square(grad) for grad in grads]

				avg_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]
				avg_square_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]

				update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])) 
					for grad_pair in zip(avg_grads, grads)]
				update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))) 
					for grad_pair in zip(avg_square_grads, grads)]
				avg_grad_updates = update_avg_grads + update_avg_square_grads

				rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
					for avg_grad_pair in zip(avg_grads, avg_square_grads)]


				rms_updates = [grad_rms_pair[0] / grad_rms_pair[1] for grad_rms_pair in zip(grads, rms)]
				train = optimizer.apply_gradients(zip(rms_updates, params))

				return tf.group(train, tf.group(*avg_grad_updates))


	def get_weights(self, shape, name):
		fan_in = np.prod(shape[0:-1])
		std = 1 / math.sqrt(fan_in)
		return tf.Variable(tf.random_uniform(shape, minval=(-std), maxval=std), name=(name + "_weights"))

	def get_biases(self, shape, name):
		fan_in = np.prod(shape[0:-1])
		std = 1 / math.sqrt(fan_in)
		return tf.Variable(tf.random_uniform([shape[-1]], minval=(-std), maxval=std), name=(name + "_biases"))

	def record_params(self, step):
		summary_string = self.sess.run(self.param_summaries)
		self.summary_writer.add_summary(summary_string, step)