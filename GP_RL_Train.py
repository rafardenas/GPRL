#GP_RL_train

import gym
import numpy as np
import matplotlib.pyplot as plt
import GPy
import scipy.integrate as scp
from scipy.optimize import minimize, NonlinearConstraint
from collections import namedtuple
from collections import deque
from numpy.random import default_rng
from config_GP import configGP
from timer import timer

import json
import numpy as np
from GP_RL_Constrained import *
import matplotlib.pyplot as plt
from GP_RL_Constrained import Env_base

np.set_printoptions(suppress=True)


params = {'CpA':30.,'CpB':60.,'CpC':20.,'CpH2SO4':35.,'T0':305.,'HRA':-6500.,'HRB':8000.,'E1A':9500./1.987,'E2A':7000./1.987,'A1':1.25,\
                 'Tr1':420.,'Tr2':400.,'CA0':4.,'A2':0.08,'UA':4.5,'N0H2S04':100.}
steps = 11
tf= 4
x0 = np.array([1,0,0,290,100])
bounds = np.array([[0,270],[298,500]])
config = configGP
config.dims_input = x0.shape[0]
#config.training_iter = 100    #lets start with 1 iter

time = timer()             
env   = Env_base(params, steps, tf, x0, bounds, config.no_controls, noisy=False)
decay_a = expexpl(config.alp_begin, config.alp_end, config.alp_rate, config.training_iter, increase=config.alp_up_anneal)
decay_b = expexpl(config.bet_begin, config.bet_end, config.bet_rate, config.training_iter)
agent = GP_agent
exp   = experiment(env, agent, config, decay_a, decay_b, UCB_beta1=config.bet_begin, UCB_beta2=10, bayes=True, constr=False, \
    disc_rew=True, two_V=True) #beta1 was 5000

time.start()

exp.training_loop()
exp.validation_loop()

time.end()

#save models
if config.save:
    exp.save_rew_models()
    exp.save_con_models()
    exp.save_con_models2()

variances = exp.get_var_data()
con_variancies = exp.get_var_con_data()
mean_rew = exp.mean_rew
viols = exp.error_vio
print("Done :)")


outputs = np.zeros((config.training_iter, steps - 1, config.dims_input+config.no_controls))
validation_data = np.zeros((config.valid_iter, steps - 1 , config.dims_input+config.no_controls+1))

for i in range(config.valid_iter):
    for j in range(steps - 1):
        validation_data[i,j,:] = exp.get_validation_data(exp.models[j])[i]

def plotting(data):
    fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))
    legend = ['$C_a$ (kmol m$^{-3}$)','$C_b$ (kmol m$^{-3}$)','$C_c$ (kmol m$^{-3}$)','$T$ (K)','$V$ (m$^3$)','$F$ (m$^3$hr$^{-1}$)','$T_a$ (K)', 'Production $C_c$ (kmol)']
    for j in range(data.shape[-1]):
        xx = data[:,:,j]
        for i in range(data.shape[0]):
            axs[j].plot(np.arange(len(xx[i,:])), xx[i,:])#, label = 'iteration #: {}'.format(str(i)))
            if j == 3:
                axs[j].plot(np.arange(len(xx[i,:])),[420 for i in range(len(xx))],c='r',linewidth=1.4,linestyle='--')
            if j == 4:
                axs[j].plot(np.arange(len(xx[i,:])),[800 for i in range(len(xx))],c='r',linewidth=1.4,linestyle='--')
            axs[j].set_ylabel(legend[j])
    #print(len(xx))
    plt.show()

def plot_variances(variances,title):
    fig, axs = plt.subplots(1, 1,figsize=(8,10))
    for i in range(variances.shape[0]):
        axs.plot(np.arange(0,variances.shape[1],1), variances[i][:], label="model {}".format(i))
    axs.set_title(title)
    axs.set_xlabel("Iterations")
    axs.set_ylabel("Variance")
    plt.legend()
    plt.show()

def plot_rew(rewards,title):
    fig, axs = plt.subplots(1, 1,figsize=(8,10))
    axs.plot(np.arange(0,len(rewards),1), rewards)
    axs.set_title(title)
    axs.set_xlabel("Tr Iterations")
    axs.set_ylabel("Mean Reward")
    plt.show()

def plot_viol(violations,title):
    fig, axs = plt.subplots(1, 1,figsize=(8,10))
    axs.plot(np.arange(0,len(violations),1), violations)
    axs.set_title(title)
    axs.set_xlabel("Tr Iterations")
    axs.set_ylabel("Violation ")
    plt.show()


plotting(validation_data)
plot_variances(variances, "Variances over time; Reward Models")
plot_variances(con_variancies, "Variances over time; Constraint 1 Models")
plot_rew(mean_rew, "Rewards over iters")
plot_viol(viols, "Violations")

#Plot of the reward at the end
#random search does not contributes to the stability of the controller
#try to iterate more and see if it converges