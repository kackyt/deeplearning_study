#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy, sys
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class Agent():

    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)

        model = Sequential()
        model.add(Flatten(input_shape=(1, n_st)))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(n_act))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)

        policy = EpsGreedyQPolicy(eps=0.1)
        dqn = DQNAgent(model=model, nb_actions=n_act, memory=memory, nb_steps_warmup=100, target_model_update=1e-2,
                            policy=policy)

        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        self.dqn = dqn

    def get_action_and_train(self, st, reward):
        return self.dqn.fit(st, reward, epochs=1, verbose=0)

    def stop_episode_and_train(self, st, reward):
        return self.dqn.fit(st, reward, epochs=1, verbose=0)

    def get_statistics(self):
        return []

    def save_model(self, model_dir):
        return 0

    def load_model(self, model_dir):
        return 0
