#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym, sys
import numpy as np

from agent_keras import Agent
from knapsack import KnapsackEnvironment
from card_environment import CardEnvironment

def main(load=False, seed=0):

    #env = KnapsackEnvironment()
    env = CardEnvironment()

    n_st = len(env.get_state())
    n_act = env.get_num_actions()

    agent = Agent(n_st, n_act, seed)

    agent.dqn.fit(env, nb_steps=20000, visualize=False, verbose=2, nb_max_episode_steps=300)
    agent.dqn.test(env, nb_episodes=1, visualize=False)
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

if __name__=="__main__":
    main()
