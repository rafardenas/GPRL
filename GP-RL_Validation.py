#Validation routine


import json
import GPy
import numpy as np
from config_GP import configGP
from GP_RL_Constrained import experiment, Env_base, expexpl, GP_agent
import matplotlib.pyplot as plt
from GP_RL_Constrained import Env_base

class validator(object):
    def __init__(self, experiment, config):
        self.exp = experiment
        self.config = config
        self.validation_data, self.variances, self.con_variancies = None, None, None
    
    def transfer_models(self):
        print("==========================Transfering models==========================")
        for i in range(len(self.exp.models)):
            mr_data = open('Models/Rew_model {}'.format(i))
            mr = json.load(mr_data)
            self.exp.models[i].core = GPy.Model.from_dict(mr)

            mc_data = open('Models/Con_model {}'.format(i))
            mc = json.load(mc_data)
            self.exp.con_models[i].core = GPy.Model.from_dict(mc)

            mc2_data = open('Models/Con_model2 {}'.format(i))
            mc2 = json.load(mc2_data)
            self.exp.con_models2[i].core = GPy.Model.from_dict(mc2)
        return
    
    def val_routine(self):
        self.transfer_models()
        self.exp.alpha = self.config.alp_begin
        self.exp.validation_loop()
        self.variances = self.exp.get_var_data()
        self.con_variancies = self.exp.get_var_con_data()
        self.validation_data = np.zeros((self.config.valid_iter, steps - 1 , self.config.dims_input+self.config.no_controls+1))

        for i in range(self.config.valid_iter):
            for j in range(steps - 1):
                self.validation_data[i,j,:] = self.exp.get_validation_data(exp.models[j])[i]
        return

    def plotting(self):
        fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))
        legend = ['$C_a$ (kmol m$^{-3}$)','$C_b$ (kmol m$^{-3}$)','$C_c$ (kmol m$^{-3}$)','$T$ (K)','$V$ (m$^3$)','$F$ (m$^3$hr$^{-1}$)','$T_a$ (K)', 'Production $C_c$ (kmol)']
        for j in range(self.validation_data.shape[-1]):
            xx = self.validation_data[:,:,j]
            for i in range(self.validation_data.shape[0]):
                axs[j].plot(np.arange(len(xx[i,:])), xx[i,:])#, label = 'iteration #: {}'.format(str(i)))
                if j == 3:
                    axs[j].plot(np.arange(len(xx[i,:])),[420 for i in range(len(xx))],c='r',linewidth=1.4,linestyle='--')
                if j == 4:
                    axs[j].plot(np.arange(len(xx[i,:])),[800 for i in range(len(xx))],c='r',linewidth=1.4,linestyle='--')
                axs[j].set_ylabel(legend[j])
        plt.show()
        return

    def plot_variances(self, variances,title):
        fig, axs = plt.subplots(1, 1,figsize=(8,10))
        for i in range(variances.shape[0]):
            axs.plot(np.arange(0,variances.shape[1],1), variances[i][:], label="model {}".format(i))
        axs.set_title(title)
        axs.set_xlabel("Iterations")
        axs.set_ylabel("Variance")
        plt.legend()
        plt.show()
        return


params = {'CpA':30.,'CpB':60.,'CpC':20.,'CpH2SO4':35.,'T0':305.,'HRA':-6500.,'HRB':8000.,'E1A':9500./1.987,'E2A':7000./1.987,'A1':1.25,\
                 'Tr1':420.,'Tr2':400.,'CA0':4.,'A2':0.08,'UA':4.5,'N0H2S04':100.}
steps = 11
tf= 4
x0 = np.array([1,0,0,290,100])
bounds = np.array([[0,270],[298,500]])
config = configGP
config.dims_input = x0.shape[0]

env   = Env_base(params, steps, tf, x0, bounds, config.no_controls, noisy=False)
decay_a = expexpl(config.alp_begin, config.alp_end, config.alp_rate, config.training_iter)
decay_b = expexpl(config.bet_begin, config.bet_end, config.bet_rate, config.training_iter)

agent = GP_agent

exp    = experiment(env, agent, config, decay_a, decay_b, UCB_beta1=config.bet_begin, UCB_beta2=None, bayes=False, constr=True, \
    disc_rew=True, two_V=True) #beta1 was 5000

val    = validator(exp, config)
val.val_routine()
#val.plotting()
#val.plot_variances(val.variances, "Variances over time; Reward Models")
#val.plot_variances(val.con_variancies, "Variances over time; Constraint 1 Models")

