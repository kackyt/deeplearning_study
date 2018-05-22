#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym, sys
import numpy as np

from agent import Agent
from knapsack import KnapsackEnvironment as Environment

def main(load=False, seed=0):

    env = Environment()

    n_st = 61
    n_act = 20

    agent = Agent(n_st, n_act, seed)
    if load:
        agent.load_model('./')

    for i_episode in range(20000):
        observation = env.reset()
        r_sum = 0
        q_list = []
        for t in range(n_act):
            state = np.array(observation).astype(np.float32).reshape((1,n_st))
            act_i, q = agent.get_action(state)
            q_list.append(q)
            action = act_i
            observation, reward, ep_end = env.step(action)
            state_dash = np.array(observation).astype(np.float32).reshape((1,n_st))
            agent.stock_experience(state, act_i, reward, state_dash, ep_end)
            agent.train()
            r_sum += reward
            if ep_end:
                break
        print("\t".join(map(str,[i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1) ,agent.step])))
        print("----state ----")
        print(env.get_state())
        agent.save_model('./')

if __name__=="__main__":
    main()
