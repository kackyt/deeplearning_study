#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

from agent import Agent
from knapsack import KnapsackEnvironment
from card_environment import CardEnvironment

def main(load=False, seed=0):
    file = open("result.csv", "a")

    # env = KnapsackEnvironment()
    env = CardEnvironment()

    n_st = len(env.get_state())
    n_act = env.get_num_actions()

    agent = Agent(n_st, n_act, seed)
    if load:
        agent.load_model('./')

    for i_episode in range(20000):
        observation = env.reset()
        r_sum = 0
        for t in range(2000):
            state = np.array(observation).astype(np.float32).reshape((1,n_st))
            action = agent.get_action_and_train(state, r_sum)
            observation, reward, ep_end = env.step(action)
            r_sum += reward
            if ep_end:
                agent.stop_episode_and_train(state, r_sum)
                break
        print('episode:', i_episode,
              'R:', r_sum,
              'statistics:', agent.get_statistics())
        print("----state ----")
        state = env.get_state()
        for s in state[1:20]:
            print('{0:>.3f} '.format(s), end="")
        print()
        for s in state[21:40]:
            print('{0:>.3f} '.format(s), end="")
        print()
        for s in state[41:60]:
            print('{0:>5} '.format(s), end="")
        print()
        agent.save_model('./')
        file.write(f'{r_sum},');
    file.close()

if __name__=="__main__":
    main()
