# This scripts use MCMC to calculate returns for all target policies provided by D4RL

import json
import os
import pickle
import pprint

from absl import app
from absl import flags
import d4rl  # pylint:disable=unused-import
import gym
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_probability as tfp
tfd = tfp.distributions
config=tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

class D4RL_Policy:
	"""D4RL policy."""

	def __init__(self, policy_file):
		with tf.io.gfile.GFile(os.path.join("gs://gresearch/deep-ope/d4rl", policy_file), 'rb') as f:
			weights = pickle.load(f)

		self.fc0_w = weights['fc0/weight']
		self.fc0_b = weights['fc0/bias']
		self.fc1_w = weights['fc1/weight']
		self.fc1_b = weights['fc1/bias']
		self.fclast_w = weights['last_fc/weight']
		self.fclast_b = weights['last_fc/bias']
		self.fclast_w_logstd = weights['last_fc_log_std/weight']
		self.fclast_b_logstd = weights['last_fc_log_std/bias']
		relu = lambda x: np.maximum(x, 0)
		self.nonlinearity = np.tanh if weights['nonlinearity'] == 'tanh' else relu

		identity = lambda x: x
		self.output_transformation = np.tanh if weights[
		'output_distribution'] == 'tanh_gaussian' else identity

	def act(self, state, noise):
		x = np.dot(self.fc0_w, state) + self.fc0_b
		x = self.nonlinearity(x)
		x = np.dot(self.fc1_w, x) + self.fc1_b
		x = self.nonlinearity(x)
		mean = np.dot(self.fclast_w, x) + self.fclast_b
		logstd = np.dot(self.fclast_w_logstd, x) + self.fclast_b_logstd

		action = self.output_transformation(mean + np.exp(logstd) * noise)
		return action, mean

if __name__ == "__main__":

	ENVs = [
		"hopper-medium-expert-v2", 
		"ant-medium-expert-v2", 
		"halfcheetah-medium-expert-v2", 
		"walker2d-medium-expert-v2"
	]

	for ENV_NAME in ENVs:


		with tf.io.gfile.GFile("../d4rl_policies.json", 'r') as f:
			policy_database = json.load(f)

		policy_metadata = [i for i in policy_database if i['task.task_names'][0].find(ENV_NAME.split("-")[0]+"-")!=-1]

		if not os.path.exists("./truth_discounted/"+ENV_NAME):
			os.makedirs("./truth_discounted/"+ENV_NAME)

		for p in policy_metadata:
			policy = D4RL_Policy(p['policy_path'])
			env = gym.make(ENV_NAME)
			all_returns = []
			for _ in range(1000):
				s = env.reset()
				returns = 0
				gamma = 0.995
				for t in range(env._max_episode_steps):  # pylint:disable=protected-access
					noise_input = np.random.randn(env.action_space.shape[0]).astype(
					np.float32)
					action, _ = policy.act(s, np.zeros_like(env.action_space.low))
					s, r, d, _ = env.step(action)
					returns += (gamma**t) * r
					if d:
						break
				all_returns += [returns]
			np.savetxt("./truth_discounted/"+ENV_NAME+"/"+p["policy_path"].split("/")[1]+".txt", np.asarray([np.mean(all_returns), np.std(all_returns)]))

