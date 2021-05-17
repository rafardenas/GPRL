#GP-RL Paper

import gym
import scipy.optimize
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
np.set_printoptions(suppress=True)


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
    def __init__(self, env, dims_input, model_no, name, config):
        self.env = env
        self.dims_input = dims_input
        self.kernel = GPy.kern.RBF(dims_input + self.env.no_controls, variance=1., lengthscale=None, ARD=True)
        self.inputs = []
        self.outputs = []
        self.core = None
        self.variance_con = None
        self.data_variance = []
        self.model_no = model_no
        self.name = name
        self.config = config
        self.curr_variance      = 0
        self.curr_con_variance  = 0
        self.valid_results = []
    
    def normalize_x(self, X):
        "Normalize X for the GPs"
        try:
            assert X.any() != np.nan, "input is nan,  not in place, but in the fly"
            if len(self.inputs) == 1:
                return np.zeros_like(X, dtype=np.float64)
            else:
                #print(self.inputs)
                #print(X)
                #print(np.mean(self.inputs, axis=0))
                #print(np.std(self.inputs, axis=0))
                #print(X - np.mean(self.inputs, axis=0))
                X = (X - np.mean(self.inputs, axis=0)) / np.std(self.inputs, axis=0)
                #print(X)
                X[np.isnan(X)] = 0
                #print(X)
        except AssertionError as error:
            print(error)
        return X

    def normalize_y(self, Y, reverse = False):
        "normalize Y for the GPs, not in place, but in the fly"
        if reverse:
            try:
                assert Y.any() != np.nan, "input is nan,  not in place, but in the fly"
                if len(self.inputs) == 1:
                    return np.zeros_like(Y, dtype=np.float64)
                else:
                    Y = (Y * np.std(self.outputs, axis=0) + np.mean(self.outputs, axis=0))
                    Y[np.isnan(Y)] = 0
            except AssertionError as error:
                print(error)
        else:
            try:
                assert Y.any() != np.nan, "input is nan"
                if len(self.outputs) == 1:
                    return np.zeros_like(Y, dtype=np.float64)
                else:
                    #print(self.outputs)
                    #print(Y)
                    #print(np.mean(self.outputs, axis=0))
                    #print(np.std(self.outputs, axis=0))
                    #print(Y - np.mean(self.outputs, axis=0))
                    Y = (Y - np.mean(self.outputs, axis=0)) / np.std(self.outputs, axis=0)
                    #print(Y)
                    if isinstance(Y, np.ndarray):
                        Y[np.isnan(Y)] = 0
                    #print(Y) 
            except AssertionError as error:
                print(error)

        return Y

    def update(self):
        #print(self.dims_input, self.env.no_controls)
        #instantiate a new kernel every time the GP is updated
        self.kernel = GPy.kern.RBF(self.dims_input + self.env.no_controls, variance=1., lengthscale=None, ARD=True)
        X = np.array(self.inputs).reshape(-1, self.dims_input + self.env.no_controls)
        Y = np.array(self.outputs).reshape(-1, 1)
        X = self.normalize_x(X)
        Y = self.normalize_y(Y)            #only normalize to init the model
        self.core = GPy.models.GPRegression(X, Y, self.kernel, normalizer=False)       #building/updating the GP
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
        if self.config.ls_lb is not None:
            self.core['rbf.lengthscale'].constrain_bounded(self.config.ls_lb,self.config.ls_ub)
        else:
            pass
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
        if np.isnan(Y):
            Y = 0
        self.outputs.append(Y)

    def get_outputs(self):
        return self.outputs

    def pop_input(self):
        self.inputs.pop()

    def pop_output(self):
        self.outputs.pop()

    def pop_var(self):
        self.data_variance.pop()
    


class experiment(object):
    def __init__(self, env, agent, config, decay_a, decay_b, UCB_beta1, UCB_beta2, bayes=True, constr=False, disc_rew=False, two_V=False):
        self.env = env
        self.config = config
        self.decay_a = decay_a
        self.decay_b = decay_b
        self.models = []
        self.con_models = []
        self.con_models2 = []
        self.two_V = two_V
        for i in range(self.env.steps): #instansiating one GP per time/control step to learn rewards
            self.models.append(agent(self.env, self.config.dims_input, i, "Rew {}".format(i), self.config))
        
        for i in range(self.env.steps): #one GP per time/control step to account for constraint violations
            self.con_models.append(agent(self.env, self.config.dims_input, i, "Constraint 1,{}".format(i), self.config))

        if two_V:
            for i in range(self.env.steps): #one GP per time/control step to account for constraint violations
                self.con_models2.append(agent(self.env, self.config.dims_input, i, "Constraint 2,{}".format(i), self.config))

            
        
        self.opt_result  = 0
        self.rew_history = np.zeros(self.env.steps)
        self.ns_history  = np.zeros((self.env.steps, self.config.dims_input))
        self.s_history   = np.zeros((self.env.steps, self.config.dims_input))
        self.a_history   = np.zeros((self.env.steps, self.config.no_controls))
        self.v_history   = np.zeros(self.env.steps)
        self.v2_history  = np.zeros(self.env.steps)

        self.alpha              = config.alp_begin
        self.UCB_beta1, self.UCB_beta2 = UCB_beta1, UCB_beta2
        self.bayes, self.constr = bayes, constr
        self.disc_rew           = disc_rew
        self.training_iter      = 0
        self.val_c              = 1

        self.variances          = np.zeros((self.config.training_iter ,self.env.steps))
        self.tr_iter            = 0
        self.v1_q               = self.config.v1_q
        self.qq                 = 0
        self.mean_rew           = []
        self.error_vio          = []
        

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

    def best_action(self, state, model, con_model, con_model2 = None, validation=False): #arg max over actions
        #max_a, max_q = self.random_search(state, model)
        #return max_a
        opt_loops = 1 #trying to cut this down? #changed 19/04
        f_max = 1e10
        if not validation:
            for i in range(opt_loops):
                sol = self.optimizer_control(model, con_model, state, con_model2)#-- To be used with scipy
                if -sol.fun < f_max:
                    f_max = -sol.fun
            
            s_a = np.hstack((state, sol.x))
            point = np.reshape(s_a, (1,-1))
            #print("point")
            #print("self.qq")
            #print(sol.x)
            #print("Predicted violation", con_model.core.predict(point)[0])
            #print("Predicted violation2", con_model2.core.predict(point)[0])
            return sol.x #-- To be used with scipy
                #print(-sol.fun, sol.x)
        elif validation:
            for i in range(5):
                self.UCB_beta1, self.UCB_beta2, self.val_c = 0, 0, 0
                sol = self.optimizer_control(model, con_model, state, con_model2)#-- To be used with scipy
                if -sol.fun < f_max:
                    f_max = -sol.fun
            return sol.x ##-- Not explored, only exploit
            

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
        #print(action_guess)
        action_guess = self.random_action()
        assert isinstance(state, np.ndarray) #state has to be ndarray

        state = state
        c_model1 = con_model
        c_model2 = con_model2

        def wrappercon1(action, state=state, c_model1=c_model1):
            s_a = np.hstack((state, action))
            point = np.reshape(s_a, (1,-1))
            vio = c_model1.core.predict(point)[0]
            return c_model1.normalize_y(vio, reverse=True).item()

        def wrappercon2(action, state=state, c_model2=c_model2):
            s_a = np.hstack((state, action))
            point = np.reshape(s_a, (1,-1))
            vio = c_model2.core.predict(point)[0]
            return c_model2.normalize_y(vio, reverse=True).item()

        if self.bayes:
            opt_result = minimize(self.wrapper_UCB, action_guess, \
                args = (state, model, con_model, con_model2), bounds = self.env.bounds, options={"maxiter":5000}) 
                #fixing states and maximizing over the actions
                #The optimization here is not "constrained", have to set the constraints explicitely?
            return opt_result
        
        elif self.constr:
            converged = False
            restarts = 0
            res = 1e6
            opt_result_temp = np.nan
            con_T = NonlinearConstraint(wrappercon1, -np.inf, 0, keep_feasible=False)   #need a wrapper for that method
            con_V = NonlinearConstraint(wrappercon2, -np.inf, 10, keep_feasible=False)  #will start with the "accepted" violation at 10
            #try to set a constraint of the variance to be also minimum = less incertainty
            cons = {con_T, con_V}
            while converged != True or restarts < 5:
                action_guess = self.random_action()
                opt_result = minimize(self.wrapper_UCB, action_guess, \
                    args = (state, model, con_model, con_model2), bounds=self.env.bounds, constraints = cons, \
                    options={"maxiter":100, "disp":False})
                converged = opt_result.success
                restarts += 1
                if opt_result.fun < res:
                    res = opt_result.fun
                    opt_result_temp = opt_result
                if restarts == 3 or converged == True:
                    break
            #print("Out in {} restarts".format(restarts))
            #print(opt_result_temp)
            if isinstance(opt_result_temp, scipy.optimize.OptimizeResult):
                return opt_result_temp
            else:
                return opt_result



    def max_q(self, model, con_model, state, con_model2 = None):
        #print("Optimizing")
        #max_a, max_q = self.random_search(state, model)
        #return max_q
        opt_loops = 1 #changed 14.04      #changed 19.04
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
            if self.bayes:
                UCB = - q.item() - pen_var.item() + np.sqrt(self.UCB_beta1) * pen
            else:
                UCB = -q.item()
        else:
            #print("X GP: {}".format(model.core.X))
            #print("Y GP: {}".format(model.core.Y))

            #print("X model: {}".format(model.inputs))
            #print("Y model: {}".format(model.outputs))
            
            #print("guess:", guess)
            s_a = np.hstack((state, guess))
            point = np.reshape(s_a, (1,-1))
            #print(point, point.shape)
            point = ((point - np.mean(model.inputs, axis=0)) / np.std(model.inputs, axis=0)).reshape(1,-1)
            point[np.isnan(point)] = 0
            #print(point, point.shape)
            q, q_var = model.core.predict(point)
            #self.qq = point
            #print(model.core[''])            #printing details of the model
            #print(q[0])
            #if q[0] != 0:
            #    print(q[0]) 
            #print("q", (q,q_var))
            pen1, pen_var1 = con_model.core.predict(point)      #pen1: Temperature constraint
            pen2, pen_var2 = con_model2.core.predict(point)      #pen2: Volume constraint
            #print("Violation in optimization normalized:", pen1)
            #print("Violation in optimization:", con_model.normalize_y(pen1, reverse=True))
            model.variance_con = pen_var1
            #print("pen2", (pen2, pen_var2))
            #pen1 = max(0, pen1)
            #pen2 = max(0, pen2)
            self.alpha     = 10 #self.decay_a.update(self.tr_iter)
            self.UCB_beta1 = 20 #self.decay_b.update(self.tr_iter)

            if self.bayes:
                UCB = - q.item() * self.alpha - np.sqrt(pen_var1.item()) * self.val_c + (self.UCB_beta1) * pen1 \
                    + np.sqrt(self.UCB_beta2) * pen2

            elif self.constr:
                UCB = q.item() * self.alpha - pen_var1.item() * self.UCB_beta1  #strictly, the "beta" term is missing a square root.
                model.curr_variance = q_var.item()
                model.curr_con_variance = pen_var1.item() 
            else:
                UCB = - q.item() 
            # print q values with and without constraints
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
            m.optimize(max_f_eval = self.config.max_eval, messages=False)
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
        exception = True                          #added 28.04
        while exception == True:                  #added 28.04
            try:                                  #added 28.04
                state, action = self.s_history[-1], self.a_history[-1]
                r = self.rew_history[-1] - self.rew_history[-2]
                ns = self.ns_history[-1]
                #print("Current state", state)
                Y = r + self.config.gamma * self.max_q((self.models[-1]), (self.con_models[-1]), ns, self.con_models2[-1])
                self.models[-1].data_variance.append(self.models[-1].curr_variance)       #collect variance of the best action taken in each time step/model
                self.con_models[-1].data_variance.append(self.models[-1].curr_con_variance)       #collect variances for each model
                self.v_history[-1] = self.violation1(ns)        #violation last state
                self.v2_history[-1] = self.violation2(ns)
                V = self.v_history[-1] + self.v_history[-2]
                V2 = self.v2_history[-1] + self.v2_history[-2]


                #print("+++++++++++++++++++++")
                #print("Model #: ", self.con_models[-1].model_no)
                #print("Violation", self.v_history[-1])
                #print(self.con_models[-1].outputs[-1])
                #print("Temperature", self.s_history[-1][3])
                #print("====================")

                self.models[-1].add_input(state, action)        #add new training inputs
                self.models[-1].add_output(Y)                   #add reward 

                self.con_models[-1].add_input(state, action)    #add training points
                self.con_models[-1].add_output(V)               #add violation
                
                self.con_models2[-1].add_input(state, action)   #adding 2nd violation (when applicable)
                self.con_models2[-1].add_output(V2)             #add violation

                
                m = self.models[-1].update()                    #re-fit GPregression
                self.models[-1].constrain_lenghtsc()
                m.optimize(max_f_eval = self.config.max_eval, messages=False)                      #constrain lengtscale
                m.optimize_restarts(self.config.no_restarts)
                #print("================== Reward model: {} =========================".format(self.models[-1].model_no))
                #print(m['rbf.variance'])
                
                con_m = self.con_models[-1].update()            #re-fit constraints-keeeper model
                self.con_models[-1].constrain_lenghtsc()        #constrain lengtscale
                con_m.optimize(max_f_eval = self.config.max_eval, messages=False)
                con_m.optimize_restarts(self.config.no_restarts)
                #print("================== Constrained model1: {} =========================".format(self.con_models[-1].model_no))
                #print(con_m[''])

                con_m2 = self.con_models2[-1].update()          #re-fit constraints-keeeper model
                self.con_models2[-1].constrain_lenghtsc()       #constrain lengtscale
                con_m2.optimize(max_f_eval = self.config.max_eval, messages=False)
                con_m2.optimize_restarts(self.config.no_restarts)
                exception = False
            
            except np.linalg.LinAlgError as e:
                exception = True
                if "not positive definite, even with jitter." in str(e):
                    self.models[-1].pop_input()     #drop new training inputs
                    self.models[-1].pop_output()

                    self.con_models[-1].pop_input()
                    self.con_models[-1].pop_output()            

                    self.con_models2[-1].pop_input()  
                    self.con_models2[-1].pop_output()

                    self.models[-1].pop_var()
                    self.con_models[-1].pop_var()
                    print("++++++++++++++++++++++++++++++++++++++", end="\n")
                    print("Not positive definite matrix :( in model 10")
                    print("++++++++++++++++++++++++++++++++++++++", end="\n")
                else:
                    raise

        #while True:
        for i in range(self.env.steps-2, -1, -1):#changed 04.03 |was range(self.env.steps-2, 0, -1)
            
            exception = True                          #added 20.04
            while exception == True:                  #added 20.04
                try:                                  #added 20.04
                    state, action = self.s_history[i], self.a_history[i]
                    r = self.rew_history[i] - self.rew_history[i-1]
                    #print("state:", state)
                    #print(i)
                    #print(self.rew_history[i])
                    #print("rew" , r)
                    ns = self.ns_history[i]
                    self.v_history[i] = self.violation1(state)     #violation in ns? || changed for state
                    self.v2_history[i] = self.violation2(state)
                    #changed on 04.03 from self.models[i] || changed on 17.03 from models[i+1] to current
                    Y = r + self.config.gamma * self.max_q((self.models[i+1]), (self.con_models[i+1]), ns, (self.con_models2[i+1]))
                    self.models[i].data_variance.append(self.models[i].curr_variance)       #collect variance of the best action taken in each time step/model
                    self.con_models[i].data_variance.append(self.models[i].curr_variance)
                    #print("Which model use to get the max Q? in the same or the next time step?")
                    #V  = self.v_history[i] + self.v_history[i + 1]    
                    V  = self.v_history[i] + self.v_history[i + 1]     #changed 29/04
                    V2 = self.v2_history[i] + self.v2_history[i + 1]   #changed 30/04

                    #if self.con_models[i].model_no == 0 or self.con_models[i].model_no == 1:
                    #print("+++++++++++++++++++++")
                    #print("Model #: ", self.con_models[i].model_no)
                    #print("Violation", self.v_history[i])
                    #print(self.con_models[i].outputs[-1])
                    #print("Temperature", self.s_history[i][3])
                    #print("====================")

                
                    self.models[i].add_input(state, action)     #add new training inputs
                    self.models[i].add_output(Y)

                    self.con_models[i].add_input(state, action) #add training points
                    self.con_models[i].add_output(V)            #add violation to model

                    self.con_models2[i].add_input(state, action)   #adding 2nd violation (when applicable)
                    self.con_models2[i].add_output(V2)               #add violation

                    m = self.models[i].update()                 #fit GPregression
                    self.models[i].constrain_lenghtsc()
                    m.optimize(max_f_eval = self.config.max_eval, messages=False)
                    m.optimize_restarts(self.config.no_restarts)
                    #print("================== Reward model: {} =========================".format(self.models[i].model_no))
                    #print(m['rbf.variance'])
                    #self.models[i].data_variance.append(m['rbf.variance'].item())       #collect variances for each model

                    con_m = self.con_models[i].update()         #re-fit constraints-keeeper model
                    self.con_models[i].constrain_lenghtsc()
                    con_m.optimize(max_f_eval = self.config.max_eval, messages=False)
                    con_m.optimize_restarts(self.config.no_restarts)
                    #print("================== Constrained model1: {} =========================".format(self.con_models[i].model_no))
                    #print(con_m[''])

                    con_m2 = self.con_models2[i].update()            #re-fit constraints-keeeper model
                    self.con_models2[i].constrain_lenghtsc()
                    con_m2.optimize(max_f_eval = self.config.max_eval, messages=False)
                    con_m2.optimize_restarts(self.config.no_restarts)

                    exception = False
                except np.linalg.LinAlgError as e:
                    exception = True
                    if "not positive definite, even with jitter." in str(e):
                        self.models[i].pop_input()     #drop new training inputs
                        self.models[i].pop_output()

                        self.con_models[i].pop_input()
                        self.con_models[i].pop_output()            

                        self.con_models2[i].pop_input()  
                        self.con_models2[i].pop_output()

                        self.models[i].pop_var()
                        self.con_models[i].pop_var()
                        print("++++++++++++++++++++++++++++++++++++++", end="\n")
                        print("Not positive definite matrix :( in model {}".format(i))
                        print("++++++++++++++++++++++++++++++++++++++", end="\n")
                    else:
                        raise
        #print(self.models[0].inputs)
        return 

    def test_pred(self):
        for i in range(self.env.steps):
            m = self.models[i]
            print("inputs", m.inputs)
            print("outputs", m.outputs)

            print("core X", m.core.X)
            print("core Y", m.core.Y)
            test = m.core.X[0] + np.random.normal(0,0.1,(m.inputs[0].shape))
            print(m.core[''])
            print("test", test)
            pred = m.core.predict(test)
            print(pred)
        return

    def new_training_step(self):
        r"The 'history' arrays are overwriten in every training iteration/step"
        self.training_iter += 1
        print("Training epoch: {}".format(self.training_iter))
        state = self.env.reset()
        tr_eps_vio_error = 0
        for i in range(self.env.steps): #one <s,a,r,ns> tuple point per time step/GP #changed 17.03 before: (self.env.steps - 1)
            action = self.best_action(state, self.models[i], self.con_models[i], self.con_models2[i])
            self.a_history[i] = action
            self.ns_history[i], self.rew_history[i], t_step = self.env.transition(state, action)
            self.s_history[i] = state   

            s_a = np.hstack((state, action))
            point = np.reshape(s_a, (1,-1))
            point = self.con_models[i].normalize_x(point)
            
            #print("Inputs GP: ", self.con_models[i].inputs) 
            #print("Outputs GP: ", self.con_models[i].outputs)
            #print("Outputs core: ", self.con_models[i].core.Y)

            #print("Real reward: ", self.rew_history[i])
            #print("Predicted reward: ", self.models[i].core.predict(point))
            #print(self.models[i].core[''])

            #print("Real state", self.ns_history[i])
            pred_vio = self.con_models[i].normalize_y(self.con_models[i].core.predict(point)[0], reverse=True).item()
            real_vio = self.violation1(self.ns_history[i])
            tr_eps_vio_error += abs(pred_vio-real_vio)
            #print("Predicted violation and variance", pred_vio)
            #print("Predicted violation de-normalized", self.con_models[i].normalize_y(pred_vio, reverse=True))
            #print("Real violation", self.violation1(self.ns_history[i]))
            #print("Real violation normalized:", self.con_models[i].normalize_y(self.violation1(self.ns_history[i])))
            state = self.ns_history[i]

        if self.training_iter == 5:  
            print("Inputs GP: ", self.models[2].inputs) 
            print("Outputs GP: ", self.models[2].outputs) 
            print(self.models[2].core[''])

        self.error_vio.append(tr_eps_vio_error / i)
        #print(self.error_vio)
        #print("+++++++++++++++++++++")
        #print("History:", self.s_history)
        #print("Ns history:", self.ns_history)
        #changed from here 03.03 and 10.03
        if self.two_V:
            self.update_models_two_const()
            self.mean_rew.append(np.mean(self.rew_history))
        else:
            self.update_models()
        return


    def violations(self, state):
        Tcon_index = int(3)
        Vcon_index = 4
        violation = (state[Tcon_index] - 420)  + (max(self.ns_history[-1][Vcon_index] - 800, 0))\
            *(420/800)
        #violation = (state[Tcon_index] - self.v1_q)  + (max(self.ns_history[-1][Vcon_index] - 800, 0))\
        #    *(self.v1_q/800)    
        #if violation > 0:
        #    violation *= self.config.v_c
        return violation

    def violation1(self, state):
        Tcon_index = int(3)
        #violation = (state[Tcon_index] - 420)
        violation = max(0, (state[Tcon_index] - self.v1_q))
        #if violation > 0:
        #    violation *= self.config.v_c
        #print("violation", violation)
        return violation #, state[Tcon_index]
           
    def violation2(self, state):
        Vcon_index = 4
        #violation = (max(0, self.ns_history[-1][Vcon_index] - 800)) * (420/800)
        violation = (max(0, self.ns_history[-1][Vcon_index] - 800)) * (self.v1_q/800)
        
        #if violation > 0:
        #    violation *= self.config.v_c
        return violation

    #######################################

    def violations_prefill(self, state):
        Tcon_index = int(3)
        Vcon_index = 4
        #violation = (state[Tcon_index] - 420) + (max(state[Vcon_index] - 800, 0))\
        #    *(420/800)
        #learning tighter
        violation = (state[Tcon_index] - self.v1_q + (max(state[Vcon_index] - 800, 0))\
            *(self.v1_q/800))
        #if violation > 0:
        #    violation *= self.config.v_c
        return violation

    def violation1_prefill(self, state):
        Tcon_index = int(3)
        #violation = (state[Tcon_index] - 420)
        violation = max(0, (state[Tcon_index] - self.v1_q))
        #if violation > 0:
        #    violation *= self.config.v_c
        return violation

    def violation2_prefill(self, state):
        Vcon_index = 4
        #violation = (max(state[Vcon_index] - 800, 0)) * (420/800)
        violation = (max(state[Vcon_index] - 800, 0)) * (self.v1_q/800)        #only leraning tighter the temperature constraint
        #if violation > 0:
            #violation *= self.config.v_c
        return violation


    def pre_filling(self):
        'Act randomly to initialise the once empty GP'
        print("-------------Starting prefilling----------------")
        for i in range(self.config.pre_filling_iters):
            state = self.env.reset()
            for i in range(self.env.steps): #fill data
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
                state = ns
                
        for i in range(self.env.steps): #update models
            m = self.models[i].update()                    #refit GPregression
            #print("GP inputs:", self.models[i].inputs)
            #print("Core inputs:", m.X)
            #print("GP outputs:", self.models[i].outputs)
            #print("Core outputs", m.Y)
            self.models[i].constrain_lenghtsc()            #constrain the lenghtscale (tweak this)
            m.optimize(max_f_eval = self.config.max_eval,messages=False) #, max_f_eval = 1000)
            m.optimize_restarts(self.config.no_restarts)

            con_m = self.con_models[i].update()            #re-fit constraints-keeeper model
            self.con_models[i].constrain_lenghtsc()
            con_m.optimize(max_f_eval = self.config.max_eval,messages=False) #, max_f_eval = 1000)
            con_m.optimize_restarts(self.config.no_restarts)

            #if self.two_V:
            con_m2 = self.con_models2[i].update()
            self.con_models2[i].constrain_lenghtsc()        #constrain/fix lenghtscale 
            con_m2.optimize(max_f_eval = self.config.max_eval,messages=False)         
            con_m2.optimize_restarts(self.config.no_restarts)

        print("=========================================")
        print("Prefilling complete, amount of data points: {}".format(len(self.models[0].inputs)))
        return
            
    
    def training_loop(self):
        self.pre_filling()
        #self.test_pred()
        for i in range(self.config.training_iter):
            self.tr_iter = self.tr_iter + 1
            print("tr_iter", self.tr_iter)
            print("alpha:", self.alpha)
            print("beta:", self.UCB_beta1)
            if not self.disc_rew:
                self.training_step()
            else:
                #print("new training")
                self.new_training_step()
            #if i % self.config.val_iter_freq == 0:
            #    self.validation_iter()
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
        self.alpha = self.config.alp_begin
        self.training_iter = 0
        for i in range(self.config.valid_iter):
            self.training_iter += 1
            print('Validation epoch: {}'.format(self.training_iter))
            state = self.env.reset()
            for i in range(self.env.steps - 1):         #control steps are 1 less than training steps
                action = self.best_action(state, self.models[i+1], self.con_models[i+1], self.con_models2[i+1], validation=True)
                ns, r, t_step = self.env.transition(state, action)
                self.models[i].add_val_result(state, action)    #add new training inputsx
                state = ns
        return

    #TO BE USED WITH COLAB AND TENSORBOARD
    #def validation_iter(self):
    #    print('Validation epoch: {}'.format(self.training_iter))
    #    state = self.env.reset()
    #    for i in range(self.env.steps - 1):         #control steps are 1 less than training steps
    #        action = self.best_action(state, self.models[i+1], self.con_models[i+1], self.con_models2[i+1], validation=True)
    #        ns, r, t_step = self.env.transition(state, action)
    #        self.writer.add_scalar("Validation Temperature", state[3] , i)
    #        state = ns
    #    return 

    def get_var_data(self):
        data = np.zeros((len(self.models), len(self.models[-1].data_variance)))
        for model in range(len(self.models)):
            for var in range(len(self.models[model].data_variance)):
                x = self.models[model].data_variance[var]
                data[model][var] = x
        return data
    
    def get_var_con_data(self):
        data = np.zeros((len(self.con_models), len(self.con_models[-1].data_variance)))
        for model in range(len(self.con_models)):
            for var in range(len(self.con_models[model].data_variance)):
                x = self.con_models[model].data_variance[var]
                data[model][var] = x
        #print("Name:", self.con_models[-1].name)
        return data
    
    def save_rew_models(self):
        i = 0
        for model in self.models:
            data = model.core.to_dict()
            json.dump(data, open("Models/Rew_model {}".format(i), 'w'))
            i += 1
        return
    
    def save_con_models(self):
        i = 0
        for model in self.con_models:
            data = model.core.to_dict()
            json.dump(data, open("Models/Con_model {}".format(i), 'w'))
            i += 1
        return

    def save_con_models2(self):
        i = 0
        for model in self.con_models2:
            data = model.core.to_dict()
            json.dump(data, open("Models/Con_model2 {}".format(i), 'w'))
            i += 1
        return


    
class expexpl(object):
    def __init__(self, eps_begin, eps_end, rate, nsteps, increase=False):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.rate = rate
        self.nsteps = nsteps
        self.increase = increase
    
    def update(self, t):
        if t < self.nsteps:
            if self.increase:
                self.epsilon = self.eps_begin * np.exp(t/self.rate)
            else:
                self.epsilon = self.eps_begin * np.exp(- t/self.rate) + self.eps_end
            
        else:
            self.epsilon = self.eps_end
        return self.epsilon

    



