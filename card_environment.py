#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

class CardEnvironment():
    '''
    カードゲーム環境。
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = [random.randint(0,2), random.randint(0,2), random.randint(0,2), random.randint(0,2), random.randint(0,2), 0]
        return self.state

    def step(self, action):
        winlose = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
        num = self.state[5]
        self.state[5] += 1
        reward = winlose[self.state[num]][action]
        print("num: %d action: %d reward: %d" % (num, action, reward))
        return self.state, reward, num == 4

    def get_state(self):
        return self.state

