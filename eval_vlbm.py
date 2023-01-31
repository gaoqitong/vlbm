import tensorflow as tf
import numpy as np
from collections import deque
import random
import gym
from gym import wrappers
from VLBM import *
import os
import tensorflow_probability as tfp
import multiprocessing as mp
import os
import d4rl
import json
import pandas as pd
import argparse
import collections
from scipy.stats import spearmanr

slim = tf.contrib.slim
rnn = tf.contrib.rnn
tfd = tfp.distributions

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-gamma", type=float, help="Set discounting factor DEFAULT=0.995", default=0.995)
parser.add_argument("-code_size", type=int, help="Set dimension of the latent space DEFAULT=16", default=16)
parser.add_argument("-env", type=str, help="Choose environment from {halfcheetah-medium-expert-v2, halfcheetah-medium-v2}. Use the other script to evaluate on Ant, Hopper, Walker2d. DEFAULT=halfcheetah-medium-expert-v2", default='halfcheetah-medium-expert-v2')
parser.add_argument("-max_episodes", type=int, help="Maximum number of episodes run for evaluation", default=50)
parser.add_argument("-path", type=str, help="Path to checkpoint folder")
# Below are some constants that would not be changed
parser.add_argument("-repeat", type=int, help="Set action repeat. Since we are training on offline trajectories, so this is not needed (always set to 1)", default=1)
parser.add_argument("-max_episode_len", type=int, help="Maximum episode length, which is always 1000 for Gym-Mujoco environments", default=1000)



def evaluate(target_policy_path):
    # Function to estimate policy returns of VLBM, using MCMC

    file_appendix = ""
    env = gym.make(rl_params['env_name'])
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    env_state_dim = env.observation_space.shape[0]
    env_action_dim = env.action_space.shape[0]
    env_action_bound = env.action_space.high
    env_state_bound = None

    graph_ope_models = tf.Graph()

    graph_ac = tf.Graph()

    with tf.Session(config=config, graph=graph_ope_models) as sess_ope_models:

        with graph_ope_models.as_default():

            ope_model = OPE_Model(
                num_branch, graph_ope_models, sess_ope_models, .001, 1000, .997, CODE_SIZE,
                env_state_dim, env_state_bound, env_action_dim, file_appendix,
                4200, RANDOM_SEED, 64, MAX_EPISODE_LEN, 1.,
                is_training=False
            )


            ope_saver = ope_model.saver

            ope_saver.restore(sess_ope_models, os.path.join(ope_path, "ope_best.ckpt"))


            d4rl_qlearning = d4rl.qlearning_dataset(env)

            obs_mean = d4rl_qlearning['observations'].mean(0).astype(np.float32)
            obs_std = d4rl_qlearning['observations'].std(0).astype(np.float32)

            rew_mean = d4rl_qlearning['rewards'].mean()
            rew_std = d4rl_qlearning['rewards'].std()

            class LearnedEnv(object):
                def __init__(self, model):

                    self.model = model

                def reset(self):
                    s0 = self.model.init_z0_s0()

                    self.obs = s0
                    return s0

                def step(self, u):
                    new_obs, reward = self.model.get_zt1_s2_r(np.reshape(u, (1, env_action_dim)))
                    self.obs = new_obs
                    self.model.update_zt()

                    return new_obs, reward, False, {}

            learned_env = LearnedEnv(ope_model)

            np.random.seed(RANDOM_SEED)
            tf.set_random_seed(RANDOM_SEED)

            ep_rewards = []
            policy = D4RL_Policy(target_policy_path)

            terminal = 0

            s = learned_env.reset()
            s = s.reshape(env_state_dim)*obs_std + obs_mean
            ep_reward = 0

            for j in range(MAX_EPISODE_LEN):

                if j % REPEAT == 0:
                    a, _ = policy.act(np.reshape(s, (env_state_dim,)), np.zeros((env_action_dim,)))
                s2, r, terminal, info = learned_env.step(a)
                r = r*rew_std + rew_mean
                s2 = s2.reshape(env_state_dim)*obs_std + obs_mean

                ep_reward += r*(GAMMA**j)

                s = s2

                if terminal or j == MAX_EPISODE_LEN-1:
                    ep_rewards += [ep_reward]

                    return ep_reward
    

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(log_device_placement=False)


    GAMMA = args.gamma
    RANDOM_SEED = args.seed
    MAX_EPISODE_LEN = args.max_episode_len
    REPEAT = args.repeat # Action repeat is not needed since we are training on offline trajectories. So it's always set to 1.
    CODE_SIZE = args.code_size
    MAX_EPISODES = args.max_episodes

    ENV = args.env

    assert "halfcheetah" in ENV, "This script only work for Halfcheetah which does not perform early termination of episodes. To train on Ant, Hopper, Walker2d, please use the other script."

    ope_path = args.path

    rl_params = {
        'env_name':ENV,
    }

    with tf.io.gfile.GFile("./d4rl_policies.json", 'r') as f:
        policy_database = json.load(f)
    policy_metadatas = [i for i in policy_database if i['task.task_names'][0].find(rl_params['env_name'].split("-")[0]+"-")!=-1]


    # Determine number of branches of VLBM    
    env = gym.make(rl_params['env_name'])
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    env_state_dim = env.observation_space.shape[0]
    env_action_dim = env.action_space.shape[0]
    env_action_bound = env.action_space.high
    env_state_bound = None

    graph_ope_models = tf.Graph()

    with graph_ope_models.as_default():
        tf.train.import_meta_graph(os.path.join(ope_path, "ope_best.ckpt.meta"))
        num_branch = np.asarray(list((set([int(v.name.split("/")[0].split("_")[-1]) for v in tf.trainable_variables() if v.name.find("Decoder_zt1_")!=-1])))).max()+1



    preds = []
    truths = []

    for i in range(11):

        target_policy_path = policy_metadatas[i]['policy_path']
        
        print("********{}********".format(policy_metadatas[i]['policy_path']))

        truths += [np.loadtxt("./truth_discounted/" + target_policy_path + ".txt")[0]]
        
        pool = mp.Pool(6)
        ep_rewards = pool.map(evaluate, [target_policy_path for _ in range(MAX_EPISODES)])
        pool.close()
        pool.join()

        preds += [np.mean(ep_rewards)]

    preds = np.asarray(preds)
    truths = np.asarray(truths)
    print ("MAE:", np.mean(np.abs((preds - truths))))

    rank, _ = spearmanr(preds, truths)
    print ("Rank:", rank)

    print("Regret:", (np.max(truths) - truths[np.argmax(preds)])/np.max(truths))













