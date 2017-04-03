from __future__ import division
from __future__ import print_function
import utils
from tqdm import tqdm
from network import *
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
from datetime import datetime
import cPickle as pickle
import cifar10

from q_network_lstm import QNetworkLSTM
startTime = datetime.now()

cifar10_upsampled, cifar10_labels_upsampled = cifar10.load_upsampled_files()
cifar10_test, cifar10_labels_test = cifar10.load_test_files()



def main():
    run_experiment_qnetwork1()


def run_experiment_qnetwork1():
    qnetwork = QNetworkLSTM(args=None, num_actions=4, scope_name="global")

    experience_replay = []
    folder = None
    #folder = 'full run qnetwork'#'2017-03-09 22:39:45.174536'
    if folder==None:
        folder = startTime.isoformat(' ')

    path = "./models_cifar/" + folder 
    if not os.path.exists(path):
        os.makedirs(path)
    experience_replay_file = path + "/experience_replay.p"

    experience_replay = []
    #with open(experience_replay_file, 'rb') as pfile:
    #    experience_replay = pickle.load(pfile)
    
    def trainNetwork(network, epochs, test_saves, experience_replay, experience_replay_file, historic_accuracies, episode, greedy_epsilon=1.0):
        saver_file = path + "/" + str(len(experience_replay))
        startTrainTime = datetime.now()
        test_acc, test_accuracies = train(network, epochs=epochs, batch_size=100, test_saves=test_saves)
        endTrainTime = datetime.now()
        delta = endTrainTime - startTrainTime
        train_time_seconds = delta.seconds + delta.microseconds/1E6
        print("train_time_seconds:", train_time_seconds)
        network.save(saver_file)
        historic_accuracies = list(historic_accuracies)
        historic_accuracies.append(test_acc)
        output = {'greedy_epsilon': greedy_epsilon, 'encoding':list(encoding), 'test_accuracy':test_acc, 'index':len(experience_replay), 'layers':layers, 'episode':episode, 'train_time_seconds':train_time_seconds, 'test_accuracies':test_accuracies, 'historic_accuracies':list(historic_accuracies)}
        experience_replay.append(output)
        with open(experience_replay_file, 'wb') as pfile:
            pickle.dump(experience_replay, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        return output

    qnetwork.save(path + "/" + "qnetwork")
    qnetwork.save(path + "/" + "qnetwork")

    greedy_epsilon = 1.0
    layers_nums = 3
    for start in range(10):
        #greedy_epsilon = max(greedy_epsilon - 0.1, 0)
        encoding = get_encoding(qnetwork, greedy_epsilon, layers_nums*4) 
        layers = decodeNetwork(encoding)
        with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
            epochs = 20000
            test_saves = 10
            output = trainNetwork(network, epochs, test_saves, experience_replay, experience_replay_file, [], episode=start, greedy_epsilon=1.0)

    skip = 3
    for layers_nums in range(6, 21, skip):
        episodes = 0
        last = [e for e in experience_replay if len(e['layers']) + skip == layers_nums]
        for dic in last:
            last_index = dic['index']
            historic_accuracies = list(dic['historic_accuracies'])
            encoding = extend_encoding(qnetwork, dic['encoding'], 1.0, layers_nums*4)
            layers = decodeNetwork(encoding)
            with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
                vs = network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
                variables = []
                for i in range(layers_nums - skip):
                    variables.extend([var for var in vs if var.name.startswith('global/'+str(i)+'/')])

                for v in variables:
                    print(v)
                network.restore_part(path + "/" + str(last_index), variables)

                epochs = 20000
                test_saves = 10
                output = trainNetwork(network, epochs, test_saves, experience_replay, experience_replay_file, historic_accuracies, episode=dic['episode'], greedy_epsilon=1.0)
            episodes += 1

        temp = [ex for ex in experience_replay if len(ex['layers']) == layers_nums]
        actions = np.array([encoding_to_one_hot(rep['encoding']) for rep in temp])
        rewards = []
        for dic in temp:
            rew = []
            last_accuracy = 0
            for acc in dic['historic_accuracies']:
                rew.extend([0]*(4*skip-1))
                rew.append(acc - last_accuracy)
                last_accuracy = acc
            rewards.append(rew)
        rewards = np.array(rewards)
        terminals = np.array([[0, 0, 0, 0]*(layers_nums-1) + [0, 0, 0, 1] for rep in temp])
        for i in range(3500):
            indices = np.random.choice(actions.shape[0], size=2)
            a = actions[indices]
            r = rewards[indices]
            t = terminals[indices]
            #if i%100==0:
            #    network.sess.run(network.update_target_variables_op)
            #    print("update")
            loss = qnetwork.train(actions=a, rewards=r, is_terminals=t)
            if i%100==0:
                print(i)
                print('loss', loss)

        qnetwork.save(path + "/" + "qnetwork")

        for i in range(5):
            greedy_epsilon = max(0.8-i*0.3, 0)
            all_encoding = get_encoding(qnetwork, greedy_epsilon, layers_nums*4) 
            for num_layers in range(3,layers_nums+1, skip):
                encoding = all_encoding[0:num_layers*4]
                #encoding = [2, 2, 1, 2, 2, 3]
                print(encoding)
                layers = decodeNetwork(encoding)
                print(layers)
                with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
                    if num_layers > 3:
                        last_index = len(experience_replay)-1

                        vs = network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
                        variables = []
                        for i in range(num_layers - skip):
                            variables.extend([var for var in vs if var.name.startswith('global/'+str(i)+'/')])

                        for v in variables:
                            print(v)
                        network.restore_part(path + "/" + str(last_index), variables)

                        historic_accuracies = list(experience_replay[last_index]['historic_accuracies'])
                    else:
                        historic_accuracies = []

                    epochs = 20000
                    test_saves = 10
                    output = trainNetwork(network, epochs, test_saves, experience_replay, experience_replay_file, historic_accuracies, episode=episodes, greedy_epsilon=greedy_epsilon)
                episodes += 1

            temp = [ex for ex in experience_replay if len(ex['layers']) == layers_nums]
            actions = np.array([encoding_to_one_hot(rep['encoding']) for rep in temp])
            rewards = []
            for dic in temp:
                rew = []
                last_accuracy = 0
                for acc in dic['historic_accuracies']:
                    rew.extend([0]*(4*skip-1))
                    rew.append(acc - last_accuracy)
                    last_accuracy = acc
                rewards.append(rew)
            rewards = np.array(rewards)
            terminals = np.array([[0, 0, 0, 0]*(layers_nums-1) + [0, 0, 0, 1] for rep in temp])
            for i in range(2500):
                indices = np.random.choice(actions.shape[0], size=2)
                a = actions[indices]
                r = rewards[indices]
                t = terminals[indices]
                #if i%100==0:
                #    network.sess.run(network.update_target_variables_op)
                #    print("update")
                loss = qnetwork.train(actions=a, rewards=r, is_terminals=t)
                if i%100==0:
                    print(i)
                    print('loss', loss)

            qnetwork.save(path + "/" + "qnetwork")



def run_experiment_qnetwork():
    qnetwork = QNetworkLSTM(args=None, num_actions=4, scope_name="global")

    experience_replay = []
    folder = None
    folder = 'full run qnetwork'#'2017-03-09 22:39:45.174536'
    if folder==None:
        folder = startTime.isoformat(' ')

    path = "./models_cifar/" + folder 
    if not os.path.exists(path):
        os.makedirs(path)
    experience_replay_file = path + "/experience_replay.p"

    with open(experience_replay_file, 'rb') as pfile:
        experience_replay = pickle.load(pfile)
    
    #print(experience_replay)
    ex_replay = []
    network_size = 15
    experience_replay_index=0
    for episode in range(15):
        accuracies = []
        print("episode:", episode)
        for layer in range(15):
            replay = experience_replay[experience_replay_index]
            assert(replay['episode'] == episode)
            assert(layer == len(replay['layers'])-1)
            encoding = replay['encoding']
            layers = decodeNetwork(encoding)
            index = replay['index']
            accuracies.append(replay['test_accuracy'])
            experience_replay_index+=1
        last_encoding = encoding
        print(last_encoding)
        actions = [[int(num == en) for num in range(4)] for en in last_encoding]
        rewards = []
        last_accuracy = 0
        for acc in accuracies:
            rewards.append(acc - last_accuracy)
            rewards.extend([0, 0, 0])
            last_accuracy = acc

        ex_replay.append({'actions': actions, 'rewards':rewards})
        assert(len(last_encoding) == len(rewards))

    print("hi")
    print(experience_replay_index)
    assert experience_replay_index == len(experience_replay)
    assert experience_replay_index//15 == len(ex_replay)
    #actions = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    #rewards = [[0.7, 0.01, -0.01, 0.1]]
    #terminals = [[0,0,0,1]]
    
    #while True:
    actions = np.array([rep['actions'] for rep in ex_replay])
    rewards = np.array([rep['rewards'] for rep in ex_replay])
    terminals = np.array([[0, 0, 0, 0]*14 + [0, 0, 0, 1] for rep in ex_replay])
    #experience_replay = [r for r in experience_replay if r['layers']]
    #terminals = [[0,1],[0,1],[0,1]]
    #rewards = [[10,-6],[10,4], [2,1]]
    #actions = [[[1, 0, 0, 0],[0, 0, 0, 1]],[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 1, 0],[0, 0, 0, 1]]]
    #actions = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    #rewards = [[0.7, 0.01, -0.01, 0.1]]
    #terminals = [[0,0,0,1]]

    feed_dict = {qnetwork.terminal_placeholder:terminals, qnetwork.rewards_placeholder:rewards, qnetwork.batchX_placeholder:actions}
    for i in range(6000):
        indices = np.random.choice(actions.shape[0], size=2)
        a = actions[indices]
        r = rewards[indices]
        t = terminals[indices]
        #if i%100==0:
        #    network.sess.run(network.update_target_variables_op)
        #    print("update")
        loss = qnetwork.train(actions=a, rewards=r, is_terminals=t)
        if i%100==0:
            print(i)
            print('loss', loss)

    greedy_epsilon = 1.0
    for i in range(15):
        greedy_epsilon = max(greedy_epsilon - 0.1, 0)
        all_encoding = get_encoding(qnetwork, greedy_epsilon, 15*4)
        accuracies = []
        for num_layers in range(1,16):
            encoding = all_encoding[0:num_layers*4]
            #encoding = [2, 2, 1, 2, 2, 3]
            print(encoding)
            layers = decodeNetwork(encoding)
            print(layers)
            with Network(input_size=[32,32,3], reshape_shape=[-1,32,32,3], num_classes=10, learning_rate=0.001, layers=layers, scope_name='global') as network:
                
                if num_layers > 1:
                    print('global/'+str(num_layers-1))
                    variables = network.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
                    variables = [var for var in variables if not var.name.startswith('global/'+str(num_layers-1))]
                    print(len(variables))
                    for v in variables:
                        print(v)
                    network.restore_part(path + "/" + str(experience_replay_index-1), variables)
                    epochs = 5000
                    test_saves = 4
                else:
                    epochs= 20000
                    test_saves = 10

                saver_file = path + "/" + str(experience_replay_index)
                startTrainTime = datetime.now()
                test_acc, test_accuracies = train(network, epochs=epochs, batch_size=100, test_saves=test_saves)
                accuracies.append(test_acc)
                endTrainTime = datetime.now()
                delta = endTrainTime - startTrainTime
                train_time_seconds = delta.seconds + delta.microseconds/1E6
                print("train_time_seconds:", train_time_seconds)
                network.save(saver_file)
                experience_replay.append({'greedy_epsilon': greedy_epsilon, 'encoding':list(encoding), 'test_accuracy':test_acc, 'index':experience_replay_index, 'layers':layers, 'episode':episode, 'train_time_seconds':train_time_seconds, 'test_accuracies':test_accuracies})
                with open(experience_replay_file, 'wb') as pfile:
                    pickle.dump(experience_replay, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            experience_replay_index += 1

        last_encoding = all_encoding
        print(last_encoding)
        actions = [[int(num == en) for num in range(4)] for en in last_encoding]
        rewards = []
        last_accuracy = 0
        for acc in accuracies:
            rewards.append(acc - last_accuracy)
            rewards.extend([0, 0, 0])
            last_accuracy = acc

        ex_replay.append({'actions': actions, 'rewards':rewards})
        assert(len(last_encoding) == len(rewards))
        assert len(experience_replay)//15 == len(ex_replay)

        actions = np.array([rep['actions'] for rep in ex_replay])
        rewards = np.array([rep['rewards'] for rep in ex_replay])
        terminals = np.array([[0, 0, 0, 0]*14 + [0, 0, 0, 1] for rep in ex_replay])
        #experience_replay = [r for r in experience_replay if r['layers']]
        #terminals = [[0,1],[0,1],[0,1]]
        #rewards = [[10,-6],[10,4], [2,1]]
        #actions = [[[1, 0, 0, 0],[0, 0, 0, 1]],[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 1, 0],[0, 0, 0, 1]]]
        #actions = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
        #rewards = [[0.7, 0.01, -0.01, 0.1]]
        #terminals = [[0,0,0,1]]

        feed_dict = {qnetwork.terminal_placeholder:terminals, qnetwork.rewards_placeholder:rewards, qnetwork.batchX_placeholder:actions}
        for i in range(3500):
            indices = np.random.choice(actions.shape[0], size=2)
            a = actions[indices]
            r = rewards[indices]
            t = terminals[indices]
            #if i%100==0:
            #    network.sess.run(network.update_target_variables_op)
            #    print("update")
            loss = qnetwork.train(actions=a, rewards=r, is_terminals=t)
            if i%100==0:
                print(i)
                print('loss', loss)



    #pred, targets, last_q, max_action = network.sess.run([network.predictions, network.targets, network.last_policy_q, network.max_action_values], feed_dict=feed_dict)
        
  #  print(i)
   # print('loss', loss)

    #print(max_action)
   # print('last_q', last_q)
   # print('pred', pred)
   # print('targets', targets)
   # q_values = network.inference(actions)
   # print(q_values)

def encoding_to_one_hot(encoding):
    return [[int(num == en) for num in range(4)] for en in encoding]

def single_step(network, greedy, encoding):
    if random.random() < greedy:
        return random.randint(0, 3)
    else:
        if len(encoding) == 0:
            qs = network.inference(np.zeros([1,0,4]))
        else:
            one_hot = encoding_to_one_hot(encoding)
            qs = network.inference([one_hot])
        return np.argmax(qs)


def extend_encoding(network, encoding, greedy, length):
    encoding = list(encoding)
    #start = network.inference(np.zeros([1,0,4]))
    #encoding.append(np.argmax(start))
    while len(encoding) < length:
        encoding.append(single_step(network, greedy, encoding))
    return encoding

def get_encoding(network, greedy, length):
    encoding = []
    return extend_encoding(network, encoding, greedy, length)

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
                test_acc, test_accuracies = train(network, epochs=20000, batch_size=100)
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