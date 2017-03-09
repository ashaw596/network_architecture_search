import utils
from tqdm import tqdm
from network import *
from tensorflow.examples.tutorials.mnist import input_data
import random
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def main():

	testEncoding = [2, 2, 1, 2, 2, 3]
	layers = decodeNetwork(testEncoding)
	print layers
	network = Network(input_size=[28*28], reshape_shape=[-1,28,28,1], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global')
	print(mnist.test)
	for i in tqdm(range(10000)):
		batch = mnist.train.next_batch(100)
		train_op, accuracy = network.train(x=batch[0], y=batch[1])
		if (i+1)%100==0:
			print(accuracy)
		if (i+1)%1000==0:
			
			test_batch = (mnist.test.images, mnist.test.labels);
			acc = network.test(x=test_batch[0], y=test_batch[1])
			print("test accuracy: ")
			print(acc)


def decodeNetwork(encoding):
	layers = []
	for layer_tuple in utils.grouper(3, encoding):
		filter_widths = [1, 3, 5, 7]
		filter_heights = [1, 3, 5, 7]
		num_filters = [24, 36, 48, 64]

		filter_widths_i, filter_heights_i, num_filters_i = layer_tuple
		
		filter_width = filter_widths[filter_widths_i]
		filter_height = filter_heights[filter_heights_i]
		num_filter = num_filters[num_filters_i]

		layers.append(ConvolutionalLayer(kernel_size=[filter_width, filter_height], stride=[1, 1], num_filters=num_filter))
	return layers

def evaluate_agent(args, agent, test_emulator, test_stats):
	step = 0
	games = 0
	reward = 0.0
	reset = test_emulator.reset()
	agent.test_state = list(list(zip(*reset))[0])
	screen = test_emulator.preprocess()
	visuals = None
	if args.watch:
		visuals = Visuals(test_emulator.get_possible_actions())

	while (step < args.test_steps) and (games < args.test_episodes):
		while not test_emulator.isGameOver() and step < args.test_steps_hardcap:
			action, q_values = agent.test_step(screen)
			results = test_emulator.run_step(action)
			screen = results[0]
			reward += results[4]

			# record stats
			if not (test_stats is None):
				test_stats.add_reward(results[4])
				if not (q_values is None):
					test_stats.add_q_values(q_values)
				# endif
			#endif

			# update visuals
			if args.watch and (not (q_values is None)):
				visuals.update(q_values)

			step +=1
		# endwhile
		games += 1
		if not (test_stats is None):
			test_stats.add_game()
		reset = test_emulator.reset()
		agent.test_state = list(list(zip(*reset))[0])

	return reward / games



def run_experiment(args, agent, test_emulator, test_stats):
	
	startBeta = 0
	endBeta = 0
	agent.run_random_exploration()

	print ("begin epochs")
	for epoch in range(1, args.epochs + 1):
		percent = float(epoch-1)/(args.epochs)
		beta = startBeta*(1-percent) + endBeta*percent
		if epoch == 1:
			agent.run_epoch(args.epoch_length - agent.random_exploration_length, epoch, beta)
		else:
			agent.run_epoch(args.epoch_length, epoch, beta)

		results = evaluate_agent(args, agent, test_emulator, test_stats)
		print("Score for epoch {0}: {1}".format(epoch, results))
		steps = 0
		if args.parallel:
			steps = agent.random_exploration_length + (agent.train_steps * args.training_frequency)
		else:
			steps = agent.total_steps

		test_stats.record(steps)
		if results >= args.saving_threshold:
			agent.save_model(epoch)
		
if __name__ == "__main__":
	main()