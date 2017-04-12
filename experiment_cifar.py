from __future__ import division
from __future__ import print_function
from tqdm import tqdm
from network import *
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
from datetime import datetime
import six.moves.cPickle as pickle
import cifar10
import utils

startTime = datetime.now()

#cifar10_upsampled, cifar10_labels_upsampled = cifar10.load_upsampled_files()
#cifar10_test, cifar10_labels_test = cifar10.load_test_files()


def main():
    test_caffe()

def test_caffe():
    path = "./test_caffe"
    for i in range(20):

        encoding = np.random.random_integers(0, 3, size=20)
        layers = decodeNetwork(encoding)
        with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.025, layers=layers, decay_steps = None, decay_rate = None, manual_learning_rate=True, scope_name='global') as network:
            caffe = network.to_caffe("network")
            with  open(path + '/' + str(i) + '.txt', "w") as text_file:
                print(caffe)
                text_file.write(caffe)

def test_single():
    #194
    encoding = [3, 2, 0, 1, 0, 3, 2, 1, 2, 3, 1, 0, 0, 1, 3, 2, 1, 2, 3, 0, 1, 2, 3, 0, 3, 2, 0, 1, 0, 1, 2, 3, 3, 2, 0, 1, 1, 0, 3, 2, 0, 1, 3, 2, 0, 3, 2, 1, 1, 2, 0, 3, 1, 3, 2, 0, 1, 3, 2, 0]
    
    #328
    #encoding = [2, 1, 3, 0, 2, 2, 3, 3, 3, 2, 3, 1, 0, 2, 3, 0, 3, 3, 3, 1, 3, 3, 3, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 3, 1, 1, 1, 1]
    layers = decodeNetwork(encoding)
    #epochs= 160000
    learning_rates = [0.025, 0.0125, 0.0001, 0.00001]
    epochs = [40, 40, 160, 60]
    #test_saves = [e//10 for e in epochs]
    path = "./models_cifar/singles"
    #with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.025, layers=layers, decay_steps = epochs//4, decay_rate = 0.1, scope_name='global') as network:
    with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.025, layers=layers, decay_steps = None, decay_rate = None, manual_learning_rate=True, scope_name='global') as network:
        print (network.to_caffe("network"))
        for learning_rate, epoch in zip(learning_rates, epochs):
            test_saves = epoch//4
            batches = epoch * 600
            saver_file = path + "/" + str("best_reinforce_2")
            #if i > 0:
            #    network.restore(path + "/" + str(i-1))
            
            #test_batch = (mnist.test.images, mnist.test.labels);
            #acc = network.test(x=test_batch[0], y=test_batch[1], batch_size=1000)
            #print("test accuracy: ")
            #print(acc)
            
            startTrainTime = datetime.now()
            network.sess.run(network.learning_rate.assign(learning_rate))
            print(learning_rate)
            print(batches)
            test_acc, test_accuracies = train(network, epochs=batches, batch_size=100, test_saves=test_saves)
            endTrainTime = datetime.now()
            delta = endTrainTime - startTrainTime
            train_time_seconds = delta.seconds + delta.microseconds/1E6
            print("train_time_seconds:", train_time_seconds)
            network.save(saver_file)
        #experience_replay.append({'encoding':list(encoding), 'test_accuracy':test_acc, 'index':i, 'layers':layers, 'episode':episode, 'train_time_seconds':train_time_seconds, 'test_accuracies':test_accuracies})
        #with open(experience_replay_file, 'wb') as pfile:
        #    pickle.dump(experience_replay, pfile, protocol=pickle.HIGHEST_PROTOCOL)
def run_experiment_finetuning():
    #test()
    experience_replay = []
    folder = None
    folder = 'full run fine_tuning'#'2017-03-09 22:39:45.174536'
    if folder==None:
        folder = startTime.isoformat(' ')

    path = "./models_cifar/" + folder 
    if not os.path.exists(path):
        os.makedirs(path)
    experience_replay_file = path + "/experience_replay.p"


    old_folder = 'full run'#'2017-03-09 22:39:45.174536'

    old_path = "./models_cifar/" + old_folder 
    if not os.path.exists(old_path):
        os.makedirs(old_path)
    old_experience_replay_file = old_path + "/experience_replay.p"
    
    with open(old_experience_replay_file, 'rb') as pfile:
        old_experience_replay = pickle.load(pfile)

    print(old_experience_replay)
    

    experience_replay = []
    i=0
    for episode in range(15):
        print("episode:", episode)
        for layer in range(15):
            print(layer)
            replay = old_experience_replay[i]
            #print(replay)
            assert(replay['episode'] == episode)
            assert(layer == len(replay['layers'])-1)
            encoding = replay['encoding']
            layers = decodeNetwork(encoding)
            index = replay['index']
            with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
                if layer > 0:
                    print('global/'+str(layer))
                    variables = network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
                    variables = [var for var in variables if not var.name.startswith('global/'+str(layer))]
                    print(len(variables))
                    for v in variables:
                        print(v)
                    network.restore_part(path + "/" + str(index-1), variables)
                    epochs = 5000
                    test_saves = 4
                else:
                    epochs= 20000
                    test_saves = 10
                saver_file = path + "/" + str(i)
                #if i > 0:
                #    network.restore(path + "/" + str(i-1))
                
                #test_batch = (mnist.test.images, mnist.test.labels);
                #acc = network.test(x=test_batch[0], y=test_batch[1], batch_size=1000)
                #print("test accuracy: ")
                #print(acc)
                
                startTrainTime = datetime.now()
                test_acc, test_accuracies = train(network, epochs=epochs, batch_size=100, test_saves=test_saves)
                endTrainTime = datetime.now()
                delta = endTrainTime - startTrainTime
                train_time_seconds = delta.seconds + delta.microseconds/1E6
                print("train_time_seconds:", train_time_seconds)
                network.save(saver_file)
                experience_replay.append({'encoding':list(encoding), 'test_accuracy':test_acc, 'index':i, 'layers':layers, 'episode':episode, 'train_time_seconds':train_time_seconds, 'test_accuracies':test_accuracies})
                with open(experience_replay_file, 'wb') as pfile:
                    pickle.dump(experience_replay, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            i += 1



def run_experiment_main():
    #test()
    experience_replay = []
    folder = None
    #folder = '2017-03-09 22:39:45.174536'
    if folder==None:
        folder = startTime.isoformat(' ')

    path = "./models_cifar/" + folder 
    if not os.path.exists(path):
        os.makedirs(path)
    experience_replay_file = path + "/experience_replay.p"
    i = 0
    for episode in range(15):
        print("episode:", episode)
        encoding = []
        for layers in range(15):
            encoding.extend(random.sample(xrange(4), 4))
            #encoding = [2, 2, 1, 2, 2, 3]
            layers = decodeNetwork(encoding)
            print(layers)
            with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
                saver_file = path + "/" + str(i)
                #if i > 0:
                #    network.restore(path + "/" + str(i-1))
                '''
                test_batch = (mnist.test.images, mnist.test.labels);
                acc = network.test(x=test_batch[0], y=test_batch[1], batch_size=1000)
                print("test accuracy: ")
                print(acc)
                '''
                startTrainTime = datetime.now()
                test_acc = train(network, epochs=20000, batch_size=100)
                endTrainTime = datetime.now()
                delta = endTrainTime - startTrainTime
                train_time_seconds = delta.seconds + delta.microseconds/1E6
                print("train_time_seconds:", train_time_seconds)
                network.save(saver_file)
                experience_replay.append({'encoding':list(encoding), 'test_accuracy':test_acc, 'index':i, 'layers':layers, 'episode':episode, 'train_time_seconds':train_time_seconds})
                with open(experience_replay_file, 'wb') as pfile:
                    pickle.dump(experience_replay, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            i+=1


def train(network, epochs=10000, batch_size=100, test_saves=10):
    test_accuracies = []
    batch_generator = cifar10.get_batch_generator(cifar10_upsampled, cifar10_labels_upsampled)
    total_train_accuracy = 0
    total_train_epochs = 0
    for i in tqdm(range(epochs)):
        batch = batch_generator.next_batch(batch_size)
        train_op, accuracy = network.train(x=batch[0], y=batch[1])
        total_train_accuracy += accuracy
        total_train_epochs += 1
        if (i+1)%(epochs//test_saves)==0:
            print(total_train_accuracy/total_train_epochs)
            acc = network.test(x=cifar10_test, y=cifar10_labels_test, batch_size=batch_size)
            test_accuracies.append(acc)
            print("test accuracy: ", acc)
            total_train_accuracy = 0
            total_train_epochs = 0
    acc = network.test(x=cifar10_test, y=cifar10_labels_test, batch_size=batch_size)
    print("test accuracy: ")
    print(acc)
    return acc, test_accuracies
    

def test():
    #testEncoding = [2, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 1]
    testEncoding = [2, 2, 1, 0, 2, 3, 3, 0]
    layers = decodeNetwork(testEncoding)
    with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
        train(network, epochs=20000, batch_size=100)

def decodeNetwork(encoding):
    layers = []
    for layer_tuple in utils.grouper(4, encoding):
        filter_widths = [1, 3, 5, 7]
        filter_heights = [1, 3, 5, 7]
        num_filters = [24, 36, 48, 64]
        strides = [1, 1, 2, 3]

        filter_widths_i, filter_heights_i, num_filters_i, strides_i = layer_tuple
        
        filter_width = filter_widths[filter_widths_i]
        filter_height = filter_heights[filter_heights_i]
        num_filter = num_filters[num_filters_i]
        stride = strides[strides_i]

        layers.append(ConvolutionalLayer(kernel_size=[filter_width, filter_height], stride=[stride, stride], num_filters=num_filter))
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