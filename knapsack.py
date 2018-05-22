#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import csv

class KnapsackEnvironment():
    '''
    ナップザック環境。
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 20
        self.max_cost = 10
        self.costs = []
        self.values = []
        self.selected = []
        self.totalcost = 0

        for i in range(self.count):
            self.costs.append(random.random())
            self.values.append(random.random())
            self.selected.append(0)

        self.totalcost = 10 # random.randint(1, sum(self.costs))
        return self.get_state()

    def get_reward(self):
        return sum([self.selected[i] * self.values[i] for i in range(len(self.selected))])

    def get_cost(self):
        return sum([self.selected[i] * self.costs[i] for i in range(len(self.selected))])

    def step(self, action):
        self.selected[action] += 1
        if self.totalcost < self.get_cost():
            self.selected[action] -= 1
            return self.get_state(), self.get_reward(), True
        return self.get_state(), 0, False

    def get_state(self):
        x = []
        x.append(self.totalcost)
        x.extend(self.costs)
        x.extend(self.values)
        x.extend(self.selected)
        return x
