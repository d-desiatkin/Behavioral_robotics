import gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from copy import deepcopy


def plot_history(history):
    plt.plot(history['av_reward'], label='average reward')
    plt.hold
    plt.title('Training History')
    plt.plot(history['best_reward'], label='best reward')
    plt.hlines(score, len(history['av_reward']) * 0.8, len(history['av_reward']), label='Evaluation score')
    plt.legend(loc='best')
    plt.show()



class Gen_alg_NN(object):
    def __init__(self):
        name = 'CartPole-v1'
        self.env = gym.make(name)
        self.max_episode_steps = gym.envs.registry.spec(name).max_episode_steps
        # number of neural networks to train
        self.population_size = 32
        # number of reward evaluations for one set of weights
        self.k_folds = 5
        # variance of initial parameters
        self.pvariance = 0.1
        # variance of perturbations
        self.ppvariance = 0.02
        # number of hidden neurons
        self.nhiddens = 5
        # the number of inputs and outputs depends on the problem
        # we assume that observations consist of vectors of continuous value
        # and that actions can be vectors of continuous values or discrete actions
        self.ninputs = self.env.observation_space.shape[0]
        if (isinstance(self.env.action_space, gym.spaces.box.Box)):
            self.noutputs = self.env.action_space.shape[0]
        else:
            self.noutputs = self.env.action_space.n

        self.weights = [{} for _ in range(self.population_size)]
        self.weights_is_sorted = False

        self.init_weights()

    
    def __del__(self):
        self.env.close()

    def init_weights(self):
        for i in range(self.population_size):
            # initialize the training parameters randomly by using a gaussian distribution with
            # average 0.0 and variance 0.1
            # biases (thresholds) are initialized to 0.0
            # first layer
            self.weights[i]['W1'] = np.random.randn(self.nhiddens, self.ninputs) * self.pvariance
            # second layer
            self.weights[i]['W2'] = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance
            # bias first layer
            self.weights[i]['b1'] = np.zeros(shape=(self.nhiddens, 1))
            # bias second layer
            self.weights[i]['b2'] = np.zeros(shape=(self.noutputs, 1))
            # reward of population sample
            self.weights[i]['reward'] = 0

    def reinit_weights(self):
        self.weights_is_sorted = False
        self.init_weights()

    def simulation(self, W1, W2, b1, b2, visualize=False):
        cur_reward = 0
        observation = self.env.reset()

        if visualize:
            self.env.render()
        
        for _ in range(self.max_episode_steps):
            # convert the observation array into a matrix with 1 column and ninputs rows
            observation.resize(self.ninputs, 1)
            # compute the netinput of the first layer of neurons
            Z1 = np.dot(W1, observation) + b1
            # compute the activation of the first layer of neurons with the tanh function
            A1 = np.tanh(Z1)
            # compute the netinput of the second layer of neurons
            Z2 = np.dot(W2, A1) + b2
            # compute the activation of the second layer of neurons with the tanh function
            A2 = np.tanh(Z2)
            # if actions are discrete we select the action corresponding to the most activated unit
            if (isinstance(self.env.action_space, gym.spaces.box.Box)):
                action = A2
            else:
                action = np.argmax(A2)

            observation, reward, done, info = self.env.step(action)

            if visualize:
                sleep(0.033)
                self.env.render()

            cur_reward+=reward

            if done:
                break

        return cur_reward

    def evaluate_weights(self, i):
        W1 = self.weights[i]['W1']
        W2 = self.weights[i]['W2']
        b1 = self.weights[i]['b1']
        b2 = self.weights[i]['b2']
        k_folds_reward = 0
        for _ in range(self.k_folds):
            k_folds_reward+=self.simulation(W1, W2, b1, b2)
        self.weights[i]['reward'] = k_folds_reward // self.k_folds

    def update_weights(self):
        self.weights_is_sorted = False
        for i in range(self.population_size//2):
            self.weights[i+self.population_size//2] = deepcopy(self.weights[i])
            
        for i in range(self.population_size//2, self.population_size):
            self.weights[i]['W1'] += np.random.randn(self.nhiddens,self.ninputs) * self.ppvariance
            self.weights[i]['W2'] += np.random.randn(self.noutputs,self.nhiddens) * self.ppvariance
            self.weights[i]['b1'] += np.random.randn(self.nhiddens, 1) * self.ppvariance
            self.weights[i]['b2'] += np.random.randn(self.noutputs, 1) * self.ppvariance

    def score(self):
        av_reward = 0
        best_reward = 0
        if not self.weights_is_sorted:
            self.weights.sort(key=lambda x: x['reward'], reverse=True)
            self.weights_is_sorted = True
        for i in range(self.population_size // 2):
            av_reward += self.weights[i]['reward']
        av_reward = av_reward // (self.population_size // 2)
        best_reward = self.weights[0]['reward']
        return av_reward, best_reward

    def train(self, unstack=False):
        history = {'av_reward': [], 'best_reward': []}
        while 1:
            for i in range(self.population_size):
                self.evaluate_weights(i)
                
            self.weights.sort(key=lambda x: x['reward'], reverse=True)
            self.weights_is_sorted = True

            av_reward, best_reward = self.score()
            history['av_reward'].append(av_reward)
            history['best_reward'].append(best_reward)
            # print('Average reward: {} Best reward: {}'.format(av_reward, best_reward))

            if av_reward == best_reward and av_reward < self.max_episode_steps // 2 and unstack:
                self.reinit_weights()

            if best_reward == self.max_episode_steps:
                break
            
            self.update_weights()
        return history

    def eval(self, enable_sim=True):
        W1 = self.weights[0]['W1']
        W2 = self.weights[0]['W2']
        b1 = self.weights[0]['b1']
        b2 = self.weights[0]['b2']
        return self.simulation(W1,W2,b1,b2,enable_sim)


model = Gen_alg_NN()
complexity = []
evaluation = []
for j in range(100):
    history = model.train()
    score = model.eval(False)
    complexity.append(len(history['best_reward']))
    evaluation.append(score)
    model.reinit_weights()
    print(j)
print("Average evaluation reward: {}".format(np.mean(evaluation)))
print("Average complexity: {}".format(np.mean(complexity)))


del model
