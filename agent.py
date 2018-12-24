#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy, sys
import numpy as np
from collections import deque

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers
import chainerrl
import random

class Agent():

    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_act = n_act
        self.model = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    n_st, n_act,
    n_hidden_layers=3, n_hidden_channels=60)
        # self.model.to_gpu(0)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.loss = 0
        self.step = 0
        self.gamma = 0.8    # 割引率
        self.mem_size = 100 # 経験メモリのサイズ 
        self.batch_size = 10 # バッチのサイズ
        self.epsilon = 1
        self.epsilon_decay = 0.0005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20
        randaction = lambda : random.randrange(0, n_act)
        self.explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon = self.epsilon, end_epsilon = self.epsilon_min, decay_steps = 1 / self.epsilon_decay, random_action_func=randaction)
        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        self.replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
        # Since observations from CartPole-v0 is numpy.float64 while
        # Chainer only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(np.float32, copy=False)
        self.agent = chainerrl.agents.DQN(self.model, self.optimizer, 
                                                self.replay_buffer, self.gamma, self.explorer, 
                                                replay_start_size = 500, update_interval = 1,
                                               target_update_interval = self.target_update_freq, phi = phi)

    def get_action_and_train(self, st, reward):
      return self.agent.act_and_train(st, reward)
    
    def stop_episode_and_train(self, st, reward):
      return self.agent.stop_episode_and_train(st, reward, True)

    def get_statistics(self):
      return self.agent.get_statistics()

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)
        self.target_model = copy.deepcopy(self.model)
