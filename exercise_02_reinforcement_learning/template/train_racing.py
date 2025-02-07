import gym
from exercise_02_reinforcement_learning.template import deepq

def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.learn(env)
    env.close()


if __name__ == '__main__':
    main()
