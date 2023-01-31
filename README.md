# Variational Latent Branching Model for Off-Policy Evaluation 

Qitong Gao, Ge Gao, Min Chi, Miroslav Pajic

Paper can be found at https://arxiv.org/abs/2301.12056. Accepted to ICLR '23. 

Contact: qitong.gao@duke.edu

If you find our work and code useful, please consider cite the paper
```
@inproceedings{
gao2022gradient,
title={Variational Latent Branching Model for Off-Policy Evaluation},
author={Qitong Gao and Ge Gao and Min Chi and Miroslav Pajic},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://arxiv.org/abs/2301.12056}
}
```

----------------------------------------------------------------------------------------

***ATTENTION***

**Some of the data and checkpoints we uploaded require to be downloaded with Git Large File Storage, i.e., `git-lfs`.**

To install `git-lfs`, follow the instructions on https://github.com/git-lfs/git-lfs.

Once it is installed, make sure to clone this repository by running

`git lfs clone https://github.com/gaoqitong/vlbm.git`

or

`git lfs clone git@github.com:gaoqitong/vlbm.git`


----------------------------------------------------------------------------------------
## Summary

This folder contain the codes for training and evaluating variational latent branching models (VLBMs). 

Specifically, we provided two examples for training and evaluating (with model checkpoints provided) VLBM 
using the halfcheetah and walker2d environments, as halfcheetah will always run through a total of 1,000 
time steps and walker2d may terminiate an episode early if the states meet specific critera.


## Environmental Setup


Mujoco version: 2.1.0

Python requirements:
	Python 3.7
	tensorflow 1.15.1
	tensorflow-probability 0.8.0
	mujoco-py 2.0.2.13
	dm-control 0.0.322773188
	gym 0.21.0
	numpy 1.21.5
	pandas 1.3.5
	d4rl 1.1


## Train VLBM

We provide two scripts for training VLBMs, i.e.,

	train_vlbm.py
	train_vlbm_for_envs_with_early_termination.py

The first one is used to train VLBM on offline datasets over Halfcheetah,
which do not involve early termination of episodes.

The second script is used for environments that consider early termination,
including Ant, Hopper, Walker2d.

To train on Halfcheetah

	python train_vlbm.py -env halfcheetah-medium-expert-v2 OR
	python train_vlbm.py -env halfcheetah-medium-v2

To train on Ant/Hopper/Walker2d
	
	python train_vlbm_for_envs_with_early_termination.py -env <ant/hopper/walker2d>-medium-expert-v2 OR
	python train_vlbm_for_envs_with_early_termination.py -env <ant/hopper/walker2d>-medium-v2

[options]

	-env 			str 		"Choose environment following <env>-<dataset>-v2"

	-no_gpu			bool		"Train w/o using GPUs"

	-gpu			int		"Select which GPU to use DEFAULT=0"

	-lr	   		float 		"Set learning rate for training VLBM DEFAULT=0.0001"

	-decay_step 		int 		"Set exponential decay step DEFAULT=1000"

	-decay_rate 		float 		"Set exponential decay rate DEFAULT=0.997"

	-max_iter 		int 		"Set max number of training iterations DEFAULT=1000"

	-seed 			int 		"Set random seed DEFAULT=2599"

	-gamma 			float 		"Set discounting factor DEFAULT=0.995"

	-batch_size 		int 		"Set minibatch size DEFAULT=64"

	-num_branch 		int 		"Set number of branches for VLBM decoder DEFAULT=10"

	-code_size 		int 		"Set dimension of the latent space DEFAULT=16"

	-beta 			float 		"Set the constant C in the objective DEFAULT=1.0"

	-val_interval 		int 		"Validation interval DEFAULT=50"



## Evaluate VLBM

Similar to training, we also two seperate scripts for environments that consider (or not) 
early terminations.

We also provide checkpoints for halfcheetah-medium-expert-v2 and walker2d-medium-expert-v2
under path "./saved_model/", for reproducibility and demonstration purposes. 


To evaluate VLBM on halfcheetah-medium-expert-v2

	python eval_vlbm.py -path ./saved_model/VLBM_halfcheetah-medium-expert-v2/ -env halfcheetah-medium-expert-v2

To evaluate VLBM on walker2d-medium-expert-v2

	python eval_vlbm_for_envs_with_early_termination.py -path ./saved_model/VLBM_walker2d-medium-expert-v2/ -env walker2d-medium-expert-v2

[options]

REQUIRED:

	-path 			str 		"Path to checkpoint folder"

OTHERS:

	-no_gpu			bool 		"Train w/o using GPUs"
	-gpu 			int 		"Select which GPU to use DEFAULT=0"
	-seed 			int 		"Set random seed"
	-gamma 			float 		"Set discounting factor DEFAULT=0.995"
	-code_size 		int 		"Set dimension of the latent space DEFAULT=16"
	-env 			str 		"Choose environment following <env>-<dataset>-v2"
	-max_episodes 		int 		"Maximum number of episodes run for evaluation"















