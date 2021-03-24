#GP-RL Paper

import gym
import numpy as np
import matplotlib.pyplot as plt
import GPy
import scipy.integrate as scp
from scipy.optimize import minimize
from collections import namedtuple
from collections import deque
from numpy.random import default_rng
from config_GP import configGP


class Env_base():
    r"""
    Base class for the environment
    Attributes:
        params(dic): parameters of the diff equations for the dynamics
        steps(int): number of discrete intervals in one episode (take it as equivalent of 'movements' in Agent's class, aka one movement per step)  
        t_f(int): lenght of the episode (unitless)
        x0(array): initial state, each item is the initial state for a variable. 
        control(dict or tuple): dictionary/tuple holding the values of the controlling variables, those variables are per se the controlling actions
        modulus(array): granularity of the states discretization (only for tabular cases)
        state_UB(array): upper bound
    """
    def __init__(self, parameters, steps, tf, x0, bounds, no_controls, noisy = False):
        self.parameters = parameters
        self.steps = steps 
        self.tf = tf
        self.x0 = x0
        self.noisy = noisy
        self.bounds = bounds
        self.no_controls = no_controls
        self.time_step = 0
        self.dt = tf/steps
    
    def model(self, t, state, control):
        params = self.parameters
        globals().update(params)
        nd = 5

        Sigma_v = [1e-4,1e-4,2e-4,0.1,0.2]

        CA  = state[0]
        CB  = state[1]
        CC  = state[2]
        T   = state[3]
        Vol = state[4] 

        F   =  control[0]
        T_a =  control[1]
            
        r1 = A1*np.exp(E1A*(1./Tr1-1./T))
        r2 = A2*np.exp(E2A*(1./Tr2-1./T))

        dCA   = -r1*CA + (CA0-CA)*(F/Vol)
        dCB   =  r1*CA/2 - r2*CB - CB*(F/Vol)
        dCC   =  3*r2*CB - CC*(F/Vol)
        dT    =  (UA*10.**4*(T_a-T) - CA0*F*CpA*(T-T0) + (HRA*(-r1*CA)+HRB*(-r2*CB\
                  ))*Vol)/((CA*CpA+CpB*CB+CpC*CC)*Vol + N0H2S04*CpH2SO4)
        dVol  =  F

        response = np.array([dCA, dCB, dCC, dT, dVol])
        if self.noisy: response += np.random.normal([0 for i in range(nd)], Sigma_v)

        return response

    def reset(self):
        state = self.x0
        self.time_step = 0
        return state

    def reward(self, state):
        return state[2] * state[4]                               #paid at every iteration
    
    def discounted_reward(self, last_state, state, last_rew):
        return state[2] * state[4] - last_state[2] * last_state[4] 


    def transition(self, state, action):
        self.time_step += 1
        ode   = scp.ode(self.model)                               # define ode, using model defined as np array
        ode.set_integrator('lsoda', nsteps=3000)                     # define integrator
        ode.set_initial_value(state, self.dt)                        # set initial value
        ode.set_f_params(action)                                     # set control action
        next_state = ode.integrate(ode.t + self.dt)                       # integrate system
        reward = self.reward(state)
        next_state = np.array(next_state)
        #reward = self.discounted_reward(state, next_state)
        return next_state, reward, self.time_step



class GP_agent():
    def __init__(self, env, dims_input):
        self.env = env
        self.dims_input = dims_input
        self.kernel = GPy.kern.RBF(dims_input + self.env.no_controls, variance=1., lengthscale=1., ARD=True)
        self.inputs = []
        self.outputs = []
        self.valid_results = []
        self.core = None
    
    def normalize_x(self, X):
        "Normalize X for the GPs"
        try:
            assert X.any() != np.nan, "input is nan,  not in place, but in the fly"
            if len(self.inputs) == 1:
                return np.zeros_like(X, dtype=np.float64)
            else:
                print(self.inputs)
                print(X)
                print(np.mean(self.inputs, axis=0))
                print(np.std(self.inputs, axis=0))
                print(X - np.mean(self.inputs, axis=0))
                X = (X - np.mean(self.inputs, axis=0)) / np.std(self.inputs, axis=0)
                print(X)
                X[np.isnan(X)] = 0
                print(X)
        except AssertionError as error:
            print(error)
        return X

    def normalize_y(self, Y):
        "normalize Y for the GPs, not in place, but in the fly"
        try:
            assert Y.any() != np.nan, "input is nan"
            if len(self.outputs) == 1:
                return np.zeros_like(Y, dtype=np.float64)
            else:
                print(self.outputs)
                print(Y)
                print(np.mean(self.outputs, axis=0))
                print(np.std(self.outputs, axis=0))
                print(Y - np.mean(self.outputs, axis=0))
                Y = (Y - np.mean(self.outputs, axis=0)) / np.std(self.outputs, axis=0)
                print(Y)
                Y[np.isnan(Y)] = 0
                print(Y)
        except AssertionError as error:
            print(error)
        return Y

    def update(self):
        #print(self.dims_input, self.env.no_controls)
        X = np.array(self.inputs).reshape(-1, self.dims_input + self.env.no_controls)
        Y = np.array(self.outputs).reshape(-1, 1)
        X, Y = self.normalize_x(X), self.normalize_y(Y)            #only normalize to init the model
        self.core = GPy.models.GPRegression(X, Y, self.kernel)     #building/updating the GP
        return self.core
        
    def add_input(self, state, action):
        "state and actions have to be lists and (since 20.03) normalized"
        state, action = list(state), list(action)
        p = np.array([*state, *action]).reshape(1,-1)         
        self.inputs.append(p)
        return

    def add_input2(self, state):#function used for another use case, don't worry
        "state and actions have to be lists"
        #state, action = list(state), list(action)
        #p = np.array([*state, *action]).reshape(1,-1)         
        self.inputs.append(state)
        return
    
    def constrain_lenghtsc(self):
        self.core['rbf.lengthscale'].constrain_bounded(15,100)
        return 

    def add_val_result(self, state, action):
        "state and actions have to be lists"
        state, action = list(state), list(action)
        p = np.array([*state, *action])         
        production = p[2] * p[4]
        p = np.hstack((p, production))
        self.valid_results.append(p)
        return
        
    def add_output(self, Y):
        "since 20.03) normalized"
        if isinstance(Y, np.ndarray):
            Y = Y.item()
        self.outputs.append(Y)

    def get_outputs(self):
        return self.outputs

    


class experiment():
    def __init__(self, env, agent, config, UCB_beta1, UCB_beta2, bayes = True, disc_rew = False, two_V = False):
        self.env = env
        self.config = config
        self.models = []
        self.con_models = []
        self.con_models2 = []
        self.two_V = two_V
        for i in range(self.env.steps): #instansiating one GP per time/control step to learn rewards
            self.models.append(agent(self.env, self.config.dims_input))
        
        for i in range(self.env.steps): #one GP per time/control step to account for constraint violations
            self.con_models.append(agent(self.env, self.config.dims_input))

        if two_V:
            for i in range(self.env.steps): #one GP per time/control step to account for constraint violations
                self.con_models2.append(agent(self.env, self.config.dims_input))

            
        
        self.opt_result  = 0
        self.rew_history = np.zeros(self.env.steps)
        self.ns_history  = np.zeros((self.env.steps, self.config.dims_input))
        self.s_history   = np.zeros((self.env.steps, self.config.dims_input))
        self.a_history   = np.zeros((self.env.steps, self.config.no_controls))
        self.v_history   = np.zeros(self.env.steps)
        self.v2_history  = np.zeros(self.env.steps)

        self.UCB_beta1, self.UCB_beta2 = UCB_beta1, UCB_beta2
        self.bayes = bayes
        self.disc_rew = disc_rew
        self.training_iter = 0


    def select_action(self, state, model, con_model, con_model2 = None, pre_filling = False):
        eps = self.config.eps
        p = np.random.uniform()
        if pre_filling: p = 0
        if p > eps:
            action = self.best_action(state, model, con_model, con_model2)
        else:
            action = self.random_action()
        in_bounds = True           #the minimizer ensures that the chosen action is in the set of feasible space
        assert in_bounds
        return action

    def best_action(self, state, model, con_model, con_model2 = None): #arg max over actions
        #max_a, max_q = self.random_search(state, model)
        #return max_a
        opt_loops = 20
        f_max = 1e10
        for i in range(opt_loops):
            sol = self.optimizer_control(model, con_model, state, con_model2)#-- To be used with scipy
            if -sol.fun < f_max:
                f_max = -sol.fun
            #print(-sol.fun, sol.x)
        return sol.x #-- To be used with scipy

    def random_action(self): #random action from the action space
        actions = np.zeros((len(self.env.bounds)))
        for i in range(len(actions)):
            actions[i] = np.random.uniform(self.env.bounds[i,0], self.env.bounds[i,1])
        actions = actions
        #print(actions[0])
        return actions

    def wrapper_pred(self, guess, state, model):
        s_a = np.hstack((state, guess))
        point = np.reshape(s_a, (1,-1)) #changed here the dims
        #print(point)
        #print("guess shape", guess.shape)
        q, q_var = model.core.predict(point) 
        #print(q)
        return -q.item() #negative to make it maximization

    def optimizer_control(self, model, con_model, state, con_model2 = None):  #fix state, opt over actions
        action_guess = self.random_action()
        #print(action_guess)
        assert isinstance(state, np.ndarray) #state has to be ndarray
        opt_result = minimize(self.wrapper_UCB, action_guess, args = (state, model, con_model, con_model2), bounds = self.env.bounds) #fixing states and maximizing over the actions
                                      #The optimization here is not constrained, have to set the boundaries explicitely?
        return opt_result



    def max_q(self, model, con_model, state, con_model2 = None):
        #print("Optimizing")
        #max_a, max_q = self.random_search(state, model)
        #return max_q
        opt_loops = 20
        f_max = 1e10
        for i in range(opt_loops):
            sol = - self.optimizer_control(model, con_model, state, con_model2).fun #-- To be used with scipy
            if sol < f_max:
                f_max = sol
        return f_max

    
    def wrapper_UCB(self, guess, state, model, con_model, con_model2 = None):
        if not self.two_V:
            s_a = np.hstack((state, guess))
            point = np.reshape(s_a, (1,-1))
            point = ((point - np.mean(model.inputs, axis=0)) / np.std(model.inputs, axis=0)).reshape(1,-1) #normalizer
            #print(point)
            q, q_var = model.core.predict(point)
            pen, pen_var = con_model.core.predict(point)
            pen = max(0, pen)
            #print(q_var.item())
            if self.bayes:
                UCB = -q.item() - pen_var.item() + np.sqrt(self.UCB_beta1) * pen
            else:
                UCB = -q.item()
        else:
            print("X GP: {}".format(model.core.X))
            print("Y GP: {}".format(model.core.Y))

            print("X model: {}".format(model.inputs))
            print("Y model: {}".format(model.outputs))

            s_a = np.hstack((state, guess))
            point = np.reshape(s_a, (1,-1))
            #print(point, point.shape)
            point = ((point - np.mean(model.inputs, axis=0)) / np.std(model.inputs, axis=0)).reshape(1,-1)
            #print("model inputs {}".format(model.inputs))
            #print(point, point.shape)
            q, q_var = model.core.predict(point)
            #print(q[0])
            #if q[0] != 0:
            #    print(q[0]) 
            print("q", (q,q_var))
            pen1, pen_var1 = con_model.core.predict(point)      #pen1: Temperature constraint
            print("pen1",(pen1, pen_var1))
            pen2, pen_var2 = con_model2.core.predict(point)      #pen2: Volume constraint
            print("pen2", (pen2, pen_var2))
            pen1 = max(0, pen1)
            pen2 = max(0, pen2)
            #print(q_var.item())
            if self.bayes:
                UCB = -q.item() - pen_var1.item() + (self.UCB_beta1) * pen1 \
                    + np.sqrt(self.UCB_beta2) * pen2 
            else:
                UCB = -q.item()

            # print q values with and without constraits
            # debbug line by line
            # step by steps

            
        return UCB


    def random_search(self, state, model):
        """Optimization routine for the optimal action given the state,
        based on random search
        model: regression model from GPy"""
        actions = np.zeros(shape=(self.config.rand_search_cand, len(self.env.bounds)))
        for i in range(actions.shape[1]):
            actions[:,i] = np.random.choice(np.linspace(self.env.bounds[i,0], self.env.bounds[i,1]),\
                size=actions.shape[0],replace=False)
        state = state.reshape(1,-1)
        #print(state.shape)
        states = np.tile(state, (10,1))
        s_a = np.concatenate((states, actions), axis=1)
        #print(s_a.shape)
        landscape = model.core.predict(s_a)[0]
        #print(landscape.shape)
        optimum = np.argmax(landscape)
        #assert np.shape(optimum)[0] != 0  
        #print("iteration completed")
        return actions[optimum][:], landscape[optimum]
        

    def training_step(self):
        print("Training")
        state = self.env.reset()
        for i in range(self.env.steps):
            action = self.best_action(state, self.models[i], self.con_models[i], self.con_models2[i])
            ns, r, t_step = self.env.transition(state, action)
            Y = r + self.config.gamma * self.max_q((self.models[i]), (self.con_models[i]), ns, (self.con_models2[i]))
            self.models[i].add_input(state, action)    #add new training inputs
            self.models[i].add_output(Y)               #add new training output
            m = self.models[i].update()                #fit GPregression
            m.optimize(messages=False)
            m.optimize_restarts(self.config.no_restarts)
            state = ns
        return

    def update_models(self):
        r"""
        Written to be able to train the GP with discounted reward
        and violations. First we run the whole process, saving the RL tuple, 
        then save those points and responses (Y) in its corresponding GP
        
        """
        state, action = self.s_history[-1], self.a_history[-1]
        r = self.rew_history[-1] - self.rew_history[-2]
        #print(len(self.con_models))
        ns = self.ns_history[-1]

        Y = r + self.config.gamma * self.max_q((self.models[-1]), (self.con_models[-1]), ns, (self.con_models2[-1]))
        self.v_history[-1] = self.violations(ns)        #violation last state
        V = self.v_history[-1]

        self.models[-1].add_input(state, action)        #add new training inputs
        self.models[-1].add_output(Y)                   #add reward 

        self.con_models[-1].add_input(state, action)    #add training points
        self.con_models[-1].add_output(V)               #add violation

        
        m = self.models[-1].update()                    #re-fit GPregression
        m.optimize(messages=False)
        m.optimize_restarts(self.config.no_restarts)
        
        con_m = self.con_models[-1].update()            #re-fit constraints-keeeper model
        con_m.optimize(messages=False)
        con_m.optimize_restarts(self.config.no_restarts)
    
        
        for i in range(self.env.steps-2, -1, -1):#changed 04.03 |was range(self.env.steps-2, 0, -1)
            state, action = self.s_history[i], self.a_history[i]
            r = self.rew_history[i] - self.rew_history[i-1]
            #print(i)
            ns = self.ns_history[i]
            self.v_history[i] = self.violations(ns)     #violation in ns
            #changed 04.03 | was self.models[i]
            Y = r + self.config.gamma * self.max_q((self.models[i+1]), (self.con_models[i+1]), ns, (self.con_models2[i+1]))
            #print("Which model use to get the max Q? in the same or the next time step?")
            V = self.v_history[i] + self.v_history[i + 1]
            
            self.models[i].add_input(state, action)     #add new training inputs
            self.models[i].add_output(Y)

            self.con_models[i].add_input(state, action) #add training points
            self.con_models[i].add_output(V)            #add violation to model

            m = self.models[i].update()                 #fit GPregression
            m.optimize(messages=False)
            m.optimize_restarts(self.config.no_restarts)

            con_m = self.con_models[i].update()         #re-fit constraints-keeeper model
            con_m.optimize(messages=False)
            con_m.optimize_restarts(self.config.no_restarts)

        #print(self.models[0].inputs)
        return 
    
        
    def update_models_two_const(self):
        r"""
        Similar to the method: update_models
        But in this case we train the GPs with discounted reward
        and TWO violations, using one GP for each of them.
        """
        state, action = self.s_history[-1], self.a_history[-1]
        r = self.rew_history[-1] - self.rew_history[-2]
        ns = self.ns_history[-1]

        Y = r + self.config.gamma * self.max_q((self.models[-1]), (self.con_models[-1]), ns, self.con_models2[-1])
        self.v_history[-1] = self.violation1(ns)        #violation last state
        self.v2_history[-1] = self.violation2(ns)
        V = self.v_history[-1]
        V2 = self.v2_history[-1]

        self.models[-1].add_input(state, action)        #add new training inputs
        self.models[-1].add_output(Y)                   #add reward 

        self.con_models[-1].add_input(state, action)    #add training points
        self.con_models[-1].add_output(V)               #add violation
        
        self.con_models2[-1].add_input(state, action)   #adding 2nd violation (when applicable)
        self.con_models2[-1].add_output(V2)             #add violation

        
        m = self.models[-1].update()                    #re-fit GPregression
        self.models[-1].constrain_lenghtsc()
        m.optimize(messages=False)                      #constrain lengtscale
        m.optimize_restarts(self.config.no_restarts)
        
        con_m = self.con_models[-1].update()            #re-fit constraints-keeeper model
        self.con_models[-1].constrain_lenghtsc()        #constrain lengtscale
        con_m.optimize(messages=False)
        con_m.optimize_restarts(self.config.no_restarts)

        con_m2 = self.con_models2[-1].update()          #re-fit constraints-keeeper model
        self.con_models2[-1].constrain_lenghtsc()       #constrain lengtscale
        con_m2.optimize(messages=False)
        con_m2.optimize_restarts(self.config.no_restarts)

        
        for i in range(self.env.steps-2, -1, -1):#changed 04.03 |was range(self.env.steps-2, 0, -1)
            state, action = self.s_history[i], self.a_history[i]
            r = self.rew_history[i] - self.rew_history[i-1]
            #print(i)
            #print(self.rew_history[i])
            #print("rew" , r)
            ns = self.ns_history[i]
            self.v_history[i] = self.violations(state)     #violation in ns? || changed for state
            self.v2_history[i] = self.violation2(state)
            #changed on 04.03 from self.models[i] || changed on 17.03 from models[i+1] to current
            Y = r + self.config.gamma * self.max_q((self.models[i]), (self.con_models[i]), ns, (self.con_models2[i]))
            #print("Which model use to get the max Q? in the same or the next time step?")
            V  = self.v_history[i] + self.v_history[i + 1]
            V2 = self.v2_history[i] + self.v2_history[i + 1]

            self.models[i].add_input(state, action)     #add new training inputs
            self.models[i].add_output(Y)

            self.con_models[i].add_input(state, action) #add training points
            self.con_models[i].add_output(V)            #add violation to model

            self.con_models2[i].add_input(state, action)   #adding 2nd violation (when applicable)
            self.con_models2[i].add_output(V2)               #add violation

            m = self.models[i].update()                 #fit GPregression
            self.models[i].constrain_lenghtsc()
            m.optimize(messages=False)
            m.optimize_restarts(self.config.no_restarts)

            con_m = self.con_models[i].update()         #re-fit constraints-keeeper model
            self.con_models[i].constrain_lenghtsc()
            con_m.optimize(messages=False)
            con_m.optimize_restarts(self.config.no_restarts)

            con_m2 = self.con_models2[i].update()            #re-fit constraints-keeeper model
            self.con_models2[i].constrain_lenghtsc()
            con_m2.optimize(messages=False)
            con_m2.optimize_restarts(self.config.no_restarts)


        #print(self.models[0].inputs)
        return 



    def new_training_step(self):
        r"The 'history' arrays are overwriten in every training iteration/step"
        self.training_iter += 1
        print("Training epoch: {}".format(self.training_iter))
        state = self.env.reset()
        for i in range(self.env.steps): #one <s,a,r,ns> tuple point per time step/GP #changed 17.03 before: (self.env.steps - 1)
            action = self.best_action(state, self.models[i], self.con_models[i], self.con_models2[i])
            self.a_history[i] = action
            self.ns_history[i], self.rew_history[i], t_step = self.env.transition(state, action)
            self.s_history[i] = state   
            state = self.ns_history[i]  
        #changed from here 03.03 and 10.03
        if self.two_V:
            self.update_models_two_const()
        else:
            self.update_models()
        return


    def violations(self, state):
        Tcon_index = int(3)
        Vcon_index = 4
        violation = (state[Tcon_index] - 420)  + (max(self.ns_history[-1][Vcon_index] - 800, 0))\
            *(420/800)
        if violation > 0:
            violation *= 3
        return violation

    def violation1(self, state):
        Tcon_index = int(3)
        violation = (state[Tcon_index] - 420)
        if violation > 0:
            violation *= 3
        return violation
           
    def violation2(self, state):
        Vcon_index = 4
        violation = (max(0, self.ns_history[-1][Vcon_index] - 800)) * (420/800)
        return violation

    #######################################

    def violations_prefill(self, state):
        Tcon_index = int(3)
        Vcon_index = 4
        violation = (state[Tcon_index] - 420) + (max(state[Vcon_index] - 800, 0))\
            *(420/800)
        if violation > 0:
            violation *= 3
        return violation

    def violation1_prefill(self, state):
        Tcon_index = int(3)
        violation = (state[Tcon_index] - 420)
        if violation > 0:
            violation *= 3
        return violation

    def violation2_prefill(self, state):
        Vcon_index = 4
        violation = (max(state[Vcon_index] - 800, 0)) * (420/800)
        #if violation > 0:
        #    violation *= 3
        return violation


    def pre_filling(self):
        'Act randomly to initialise the once empty GP'
        for i in range(self.config.pre_filling_iters):
            state = self.env.reset()
            for i in range(self.env.steps):
                action = self.select_action(state, self.models[i], self.con_models[i], con_model2=self.con_models2[i], pre_filling=True)
                ns, r, t_step = self.env.transition(state, action)
                Y = r
                if not self.two_V:
                    V = self.violations_prefill(ns)
                else:
                    V = self.violation1_prefill(ns)
                    V2 = self.violation2_prefill(ns)
                
                self.models[i].add_input(state, action)        #add new training inputs
                self.models[i].add_output(Y)                   #add new training output

                self.con_models[i].add_input(state, action)    #add new training inputs
                self.con_models2[i].add_input(state, action)   #add new training inputs
                self.con_models[i].add_output(V)               #add new training output
                self.con_models2[i].add_output(V2)
                

                m = self.models[i].update()                    #refit GPregression
                print(self.models[i].inputs)
                print(m.X)
                print(self.models[i].outputs)
                print(m.Y)
                self.models[i].constrain_lenghtsc()            #constrain the lenghtscale (tweak this)
                m.optimize(messages=False) #, max_f_eval = 1000)
                m.optimize_restarts(self.config.no_restarts)

                con_m = self.con_models[i].update()            #re-fit constraints-keeeper model
                self.con_models[i].constrain_lenghtsc()
                con_m.optimize(messages=False) #, max_f_eval = 1000)
                con_m.optimize_restarts(self.config.no_restarts)

                #if self.two_V:
                con_m2 = self.con_models2[i].update()
                self.con_models2[i].constrain_lenghtsc()        #constrain/fix lenghtscale 
                con_m2.optimize(messages=False)         
                con_m2.optimize_restarts(self.config.no_restarts)

                state = ns
        print("Prefilling complete, amount of data points: {}".format(len(self.models[0].inputs)))
        return
            
    
    def training_loop(self):
        self.pre_filling()
        for i in range(config.training_iter):
            #raise AssertionError
            if not self.disc_rew:
                self.training_step()
            else:
                #print("new training")
                self.new_training_step()
        return

    def get_train_inputs(self, model):
        return model.inputs[self.config.pre_filling_iters::]

    def get_validation_data(self, model):
        return model.valid_results

    def get_train_outputs(self, model):
        return model.outputs[self.config.pre_filling_iters::]

    def get_trained_models(self):
        return self.models
    
    def validation_loop(self):
        self.training_iter = 0
        for i in range(self.config.valid_iter):
            self.training_iter += 1
            print('Validation epoch: {}'.format(self.training_iter))
            state = self.env.reset()
            for i in range(self.env.steps - 1):         #control steps are 1 less than training steps
                action = self.best_action(state, self.models[i+1], self.con_models[i+1], self.con_models2[i+1])
                ns, r, t_step = self.env.transition(state, action)
                self.models[i].add_val_result(state, action)    #add new training inputs
                state = ns
        return
    
params = {'CpA':30.,'CpB':60.,'CpC':20.,'CpH2SO4':35.,'T0':305.,'HRA':-6500.,'HRB':8000.,'E1A':9500./1.987,'E2A':7000./1.987,'A1':1.25,\
                 'Tr1':420.,'Tr2':400.,'CA0':4.,'A2':0.08,'UA':4.5,'N0H2S04':100.}
steps = 11
tf= 4
x0 = np.array([1,0,0,290,100])
bounds = np.array([[0,270],[298,500]])
config = configGP
config.dims_input = x0.shape[0]
config.training_iter = 10 #lets start with 1 iter

            
env   = Env_base(params, steps, tf, x0, bounds, config.no_controls, noisy=False)
#agent = GP_agent(env, config.input_dim)
agent = GP_agent
exp   = experiment(env, agent, config, UCB_beta1=2000, UCB_beta2=100, bayes=True,\
    disc_rew=True, two_V=True)
exp.training_loop()
exp.validation_loop()


"""
#trial = GP_agent(env, config.dims_input)
#print(trial.core)
inns = np.array([[8.67214319e-01, 1.03592312e+00, 4.01511313e-01, 3.96219817e+02,
  3.78183804e+02, 1.41935918e+02, 4.37589994e+02],
 [2.82460912e-01, 1.25708208e+00, 4.84650221e-01, 5.17531204e+02,
  3.40802688e+02, 9.20412832e+01, 4.91731495e+02],
 [2.37472379e+00, 2.66660773e-01, 1.82170060e-02, 3.49939224e+02,
  2.77826880e+02, 4.27925666e+01, 3.10711971e+02],
 [2.76691289e+00, 2.42538352e-01, 9.82571870e-03, 3.94765185e+02,
  4.04607312e+02, 6.77096436e+00, 3.94707595e+02],
 [5.22617295e-01, 1.05682430e+00, 5.89144903e-01, 4.76123839e+02,
  3.08969206e+02, 3.02658147e+01, 4.02229175e+02]])

#print(inns.shape, "\n")
outts = np.array([[151.84507593],
    [7.17009794], #165.17009794]
    [  5.06117393],
    [  3.97555763],
    [4.02763254]]) #182.02763254
trial.add_input2(inns)
trial.add_output(outts)
m = trial.update()
m.optimize()
#print(m)
test = np.array([2.37472379e+00,1.25708208e+00,5.89144903e-01,4.76123839e+02,3.78183804e+02,
3.02658147e+01,4.02229175e+02], dtype=np.float64).reshape(1,-1)
prd = m.predict(test)
#print(prd)

ker = GPy.kern.RBF(input_dim=7,variance=1., lengthscale=1., ARD=True)
m2 = GPy.models.GPRegression(inns,outts,ker, noise_var=1.)
m2.optimize(messages=False)
m2.optimize_restarts(3, messages=False)
prd2 = m2.predict(test)
#print(prd2)
#print(m2.X)
"""



#exp2   = experiment(env, agent, config, bayes=False)
#exp2.training_loop()
#exp2.validation_loop()

outputs = np.zeros((config.training_iter, steps - 1, config.dims_input+config.no_controls))
validation_data = np.zeros((config.valid_iter, steps - 1 , config.dims_input+config.no_controls+1))
#validation_data_2 = np.zeros((config.valid_iter, steps, config.dims_input+config.no_controls+1))

#for i in range(config.training_iter):
#    for j in range(steps - 1):
#        outputs[i,j,:] = exp.get_train_inputs(exp.models[j])[i]

#validating experiment

for i in range(config.valid_iter):
    for j in range(steps - 1):
        validation_data[i,j,:] = exp.get_validation_data(exp.models[j])[i]
#        validation_data_2[i,j,:] = exp2.get_validation_data(exp2.models[j])[i]



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

#print(np.mean(exp.models[-1].inputs, axis=0))
plotting(validation_data)
#plotting(validation_data_2)

#Plot of the reward at the end
#random search does not contributes to the stability of the controller
#try to iterate more and see if it converges
