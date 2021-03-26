import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as scp
import GPy 
import time
from EvolutionaryAlgorithm import * 
from pylab import *
import multiprocessing as mp 
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from scipy.optimize import minimize
import imageio 
import os 

from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

end = False


def LHS(p,bounds):
    '''
    takes a number of samples and bounds 
    returns LHS within these bounds
    '''
    sample = np.zeros((p,len(bounds)))
    for i in range(len(bounds)):
        sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
        np.random.shuffle(sample[:,i])
    return sample 

def rand(p,bounds):
    '''
    takes a number of samples and bounds 
    returns LHS within these bounds
    '''
    sample = np.zeros((p,len(bounds)))
    for i in range(len(bounds)):
        sample[:,i] = np.random.uniform(bounds[i,0],bounds[i,1],p)
    return sample 




def model(x, t, u):
    modpar    = {'CpA':30.,'CpB':60.,'CpC':20.,'CpH2SO4':35.,'T0':305.,'HRA':-6500.,'HRB':8000.,'E1A':9500./1.987,'E2A':7000./1.987,'A1':1.25,\
                 'Tr1':420.,'Tr2':400.,'CA0':4.,'A2':0.08,'UA':4.5,'N0H2S04':100.}
    globals().update(modpar)
    nd = 5

    Sigma_v = [1e-4,1e-4,2e-4,0.1,0.2]

    CA  = x[0]
    CB  = x[1]
    CC  = x[2]
    T   = x[3]
    Vol = x[4] 

    F   =  u[0]
    T_a =  u[1]
        
    r1 = A1*np.exp(E1A*(1./Tr1-1./T))
    r2 = A2*np.exp(E2A*(1./Tr2-1./T))

    dCA   = -r1*CA + (CA0-CA)*(F/Vol)
    dCB   =  r1*CA/2 - r2*CB - CB*(F/Vol)
    dCC   =  3*r2*CB - CC*(F/Vol)
    dT    =  (UA*10.**4*(T_a-T) - CA0*F*CpA*(T-T0) + (HRA*(-r1*CA)+HRB*(-r2*CB\
    ))*Vol)/((CA*CpA+CpB*CB+CpC*CC)*Vol + N0H2S04*CpH2SO4)
    dVol  =  F

    ODEeq =  np.array([dCA,dCB,dCC,dT,dVol])
    ODEeq += np.random.normal([0 for i in range(nd)],Sigma_v)

    return ODEeq

def RK4_sub(dx,t0,tn,x0,n_sub,u):
    n = len(u)
    h = (tn-t0)/(n*n_sub)             # calculating time step
    t = np.linspace(t0,tn,(n*n_sub)+1)    # creating time vector
    x = np.zeros(((n*n_sub)+1,len(x0)))   # pre-allocating solution memory
    x[0,:] = x0                       # intial conditions
    index = np.linspace(0,n*n_sub,n+1)
    for i in range(1,(n*n_sub)+1):        # calculating RK4 values
        u_index = int((i-1)/n_sub) 
        k1 = h * dx(x[i-1,:],t[i-1],u[u_index,:])
        k2 = h * dx(x[i-1,:]+(k1/2),t[i-1]+(h/2),u[u_index,:])
        k3 = h * dx(x[i-1,:]+(k2/2),t[i-1]+(h/2),u[u_index,:])
        k4 = h * dx(x[i-1,:]+k3,t[i-1]+h,u[u_index,:])
        x[i,:] = x[i-1,:] + (1/6)*(k1+2*k2+2*k3+k4) # Solution update
        
    x = x[index.astype(int)]
    t = t[index.astype(int)]

    return x ,t

def RK4_sub_plot(dx,t0,tn,x0,n_sub,u):
    n = len(u)
    h = (tn-t0)/(n*n_sub)             # calculating time step
    t = np.linspace(t0,tn,(n*n_sub)+1)    # creating time vector
    x = np.zeros(((n*n_sub)+1,len(x0)))   # pre-allocating solution memory
    x[0,:] = x0                       # intial conditions
    index = np.linspace(0,n*n_sub,n+1)
    for i in range(1,(n*n_sub)+1):        # calculating RK4 values
        u_index = int((i-1)/n_sub) 
        k1 = h * dx(x[i-1,:],t[i-1],u[u_index,:])
        k2 = h * dx(x[i-1,:]+(k1/2),t[i-1]+(h/2),u[u_index,:])
        k3 = h * dx(x[i-1,:]+(k2/2),t[i-1]+(h/2),u[u_index,:])
        k4 = h * dx(x[i-1,:]+k3,t[i-1]+h,u[u_index,:])
        x[i,:] = x[i-1,:] + (1/6)*(k1+2*k2+2*k3+k4) # Solution update
    return x, t


def q_eval(x_sub,gam): #kinda the reward function
    index = 2 
    q_vec = np.zeros(len(x_sub)-1)
    q_vec[-1] = (x_sub[-1,index]*x_sub[-1,4]) - (x_sub[-2,index]*x_sub[-2,4])
    for i in reversed(range(len(q_vec)-1)): #discounted reward 
        q_vec[i] = (x_sub[i+1,index]*x_sub[i+1,4]) - (x_sub[i,index]*x_sub[i,4]) + (gam * q_vec[i+1])
    return q_vec 

def q_con_eval(x_sub): #evaluating deviation from constraints, temperature and volume wise
    Tcon_index = int(3)
    Vcon_index = 4
    con_vec = np.zeros(len(x_sub)-1)

    con_vec[-1] = (x_sub[-1,Tcon_index]-420) + (max(x_sub[-1,Vcon_index]-800,0)*(420/800))
    if con_vec[-1] > 0:
        con_vec *= 3

    for i in reversed(range(len(con_vec)-1)): #we do it backwards to sort of back-propagate the info
        penalty_addition = x_sub[i,Tcon_index]-420 + (max(x_sub[-1,Vcon_index]-800,0)*(420/800))
        if penalty_addition > 0:
            penalty_addition *= 3 
        con_vec[i] = penalty_addition + con_vec[i+1]
    return con_vec 

def q_call(u,m,m_con,x,mean,std):
    input = np.concatenate((x,u),axis=0)
    input = (input-mean)/std
    for k in range(len(input)):
        if input[k] != input[k]:
            input[k] = 0

    q,q_var = m.predict(np.array([input]))
    pen,pen_var = m_con.predict(np.array([input]))
    #pen += 0.5*pen_var # BACKOFF 
    pen = max(pen,0)                                        #how much the constraints are violated
    multi = 100
    if bayes == True:
        cost = -q.item() - q_var.item() + pen * multi       # Upper Confidence bound
        return cost                                         # intuitively,with the later UCB, when we exploit, we take the max reward (given by the GP that are leraning rewards)
                                                            # when we explore, we take favor points where the uncertainty of the 'violation' function is the highest
    if bayes == False:
        return -q.item()  +  pen * multi
 
def q_opt(m,m_con,x,bounds,mean,std): #optimization loop
    restarts = 20
    f_max = 100000000000
    init_points =  (restarts,bounds)
    u_opt = np.zeros(len(bounds))
    for i in range(restarts):
        sol = minimize(q_call,x0=init_points[i,:],args=(m,m_con,x,mean,std),bounds=bounds)
        if sol.fun < f_max:
            f_max = sol.fun 
            u_opt = sol.x
    return u_opt


def q_controller(m,m_con,gam,information_mat):
    t = 0 
    x_store = np.copy(np.array([x0]))                                      #starting from the first state
    mean = np.mean(information_mat[:,0,:-1],axis=0)
    std = np.std(information_mat[:,0,:-1],axis=0)
    u_opt = q_opt(m[str(0)],m_con[str(0)],x0,bounds,mean,std)              #optimal control action
    u_store = np.array([u_opt])
    x_new,t_new = RK4_sub(model,t,t+0.4,x0,10,np.array([u_opt]))           #integrating the model with the optimal response
    x_new = x_new[-1,:]                                                    #select last state all the columns
    x_store = np.append(x_store,[x_new],axis=0)                            
    for i in range(1,ctrl_steps):                                          #starting the control loop (loop for n-control iterations)
        mean = np.mean(information_mat[:,i,:-1],axis=0)                    
        std = np.std(information_mat[:,i,:-1],axis=0)                       
        u_opt = q_opt(m[str(i)],m_con[str(i)],x_new,bounds,mean,std)    
        u_store = np.append(u_store,[u_opt],axis=0)
        x_new,t_new = RK4_sub(model,t,t+0.4,x_new,10,np.array([u_opt]))
        x_new = x_new[-1,:]
        x_store = np.append(x_store,[x_new],axis=0)
    q_vec = q_eval(x_store,gam)
    con_vec = q_con_eval(x_store)
    q_vec = np.reshape(q_vec,(len(q_vec),1))
    x_store = x_store[:-1,:]
    information_mat = np.concatenate((x_store,u_store,q_vec),axis=1)
    return information_mat,con_vec


def rmv_clostest(m,information_mat,con_store):
    info_copy = np.zeros((len(information_mat[:,0,0])-1,ctrl_steps,len(information_mat[0,0,:])))
    con_copy = np.zeros((len(con_store[:,0,0])-1,ctrl_steps,len(con_store[0,0,:])))

    for i in range(ctrl_steps):
        k = m[str(i)].kern.K(information_mat[:,i,:-1])
        for j in range(len(k)):
            k[j,j] = 0 
        min_dist = np.zeros(len(k))
        min_arg = np.zeros(len(k))
        for t in range(len(k)):
            min_dist[t] = np.max(k[t,:])
            min_arg[t] = np.argmax(k[t,:])
        closest1 = int(np.argmax(min_dist))
        closest2 = int(min_arg[closest1])
        closest = [closest1,closest2]
        closest_vals = [information_mat[closest1,i,-1],information_mat[closest2,i,-1]]
        min_arg = np.argmin(closest_vals)
        worst = closest[min_arg]
        info_copy[:,i,:] = np.delete(information_mat[:,i,:],worst,axis=0)
        con_copy[:,i,:] = np.delete(con_store[:,i,:],worst,axis=0)
    return info_copy,con_copy 

def GP_train(information_mat): #train with rewards
    for i in range(ctrl_steps):
        inputs = information_mat[:,i,:-1]
        outputs = information_mat[:,i,-1]
        outputs = np.reshape(outputs,(len(outputs),1))
        outputs = (outputs-np.mean(outputs))/np.std(outputs)
        if outputs[0,0] != outputs[0,0]:
            outputs[:,0] = 0 
        in_mean = np.mean(inputs,axis=0)
        in_std = np.std(inputs,axis=0)
        inputs = (inputs-in_mean)/ in_std 
        #outputs = outputs/np.max(outputs)
        # max_inputs = np.max(inputs,axis=0)
        # inputs = inputs/max_inputs
        for k in range(len(inputs[0,:])):
            if inputs[0,k] != inputs[0,k]:
                inputs[:,k] = [0 for j in range(len(inputs[:,0]))]
        kernel = GPy.kern.RBF(len(inputs[0,:]), ARD=True,variance=1.)
        #kernel = GPy.kern.Linear(len(inputs[0,:]), ARD=True)
        m[str(i)] = GPy.models.GPRegression(inputs,outputs,kernel)
        m[str(i)].optimize(messages=False)
        m[str(i)].optimize_restarts(optimise_restarts)
        #m[str(i)].param_array[0] = 0
    return m

def GP_train_con(information_mat,con_store): #train with vonstraint violations
    for i in range(ctrl_steps):
        inputs = information_mat[:,i,:-1]
        outputs = con_store[:,i,:]
        outputs = np.reshape(outputs,(len(outputs),1))
        outputs = (outputs)/np.std(outputs)
        in_mean = np.mean(inputs,axis=0)
        in_std = np.std(inputs,axis=0)
        inputs = (inputs-in_mean)/ in_std 
        #outputs = outputs/np.max(outputs)
        # max_inputs = np.max(inputs,axis=0)
        # inputs = inputs/max_inputs
        if outputs[0,0] != outputs[0,0]:
            outputs[:,0] = 0 
        for k in range(len(inputs[0,:])): 
            if inputs[0,k] != inputs[0,k]:
                inputs[:,k] = [0 for j in range(len(inputs[:,0]))]
        kernel = GPy.kern.RBF(len(inputs[0,:]), ARD=True,variance=1.)
        #kernel = GPy.kern.Linear(len(inputs[0,:]), ARD=True)
        m_con[str(i)] = GPy.models.GPRegression(inputs,outputs,kernel)
        m_con[str(i)].optimize(messages=False)
        m_con[str(i)].optimize_restarts(optimise_restarts)
        #m[str(i)].param_array[0] = 0
    return m_con


optimise_restarts = 0
initial_sample_runs = 20
bayes = True
remove = False
steps = 11
ctrl_steps = steps - 1
sub_steps = 100
bounds = np.array([[0,270],[298,500]])
x0 = np.array([1,0,0,290,100])
legend = ['$C_a$ (kmol m$^{-3}$)','$C_b$ (kmol m$^{-3}$)','$C_c$ (kmol m$^{-3}$)','$T$ (K)','$V$ (m$^3$)','$F$ (m$^3$hr$^{-1}$)','$T_a$ (K)']
t0 = 0 
tf = 4
t = np.linspace(t0,tf,ctrl_steps)
overall_runs = 1
its = 100
remove_num = 100
v_fin = False



end_reward = np.zeros((overall_runs,its))
total_violation = np.zeros((overall_runs,its))


for overall_it in range(overall_runs):

    u = LHS(ctrl_steps,bounds)
    x_sub,t_sub = RK4_sub(model,t0,tf,x0,sub_steps,u)
    q_vec = q_eval(x_sub,1)
    con_vec = q_con_eval(x_sub)

    q_vec = np.reshape(q_vec,(len(q_vec),1))
    con_vec = np.reshape(con_vec,(len(con_vec),1))

    x_sub = x_sub[:-1,:]
    information_mat = [np.concatenate((x_sub,u,q_vec),axis=1)]

    con_store = [con_vec]

    for i in range(initial_sample_runs):
        u = LHS(ctrl_steps,bounds)
        #u = u_store[i,:,:].T
        x_sub,t_sub = RK4_sub(model,t0,tf,x0,sub_steps,u)
        q_vec = q_eval(x_sub,1)
        con_vec = q_con_eval(x_sub)

        q_vec = np.reshape(q_vec,(len(q_vec),1))
        con_vec = np.reshape(con_vec,(len(con_vec),1))

        x_sub = x_sub[:-1,:]
        information_mat_add = np.concatenate((x_sub,u,q_vec),axis=1)
        information_mat = np.append(information_mat,[information_mat_add],axis=0)
        con_store = np.append(con_store,[con_vec],axis=0)



    m = {}
    m_con = {}

    m = GP_train(information_mat)
    m_con = GP_train_con(information_mat,con_store)

    t = np.linspace(t0,tf,steps)


    for iterations in tqdm(range(its)):

        information_mat_add,con_vec = q_controller(m,m_con,1,information_mat)
        con_vec = np.reshape(con_vec,(len(con_vec),1))
        information_mat = np.append(information_mat,[information_mat_add],axis=0)
        con_store = np.append(con_store,[con_vec],axis=0)

        m = GP_train(information_mat)
        m_con = GP_train_con(information_mat,con_store)
            
        if remove == True:
            if len(information_mat[:,0,0]) > remove_num:
                information_mat,con_store = rmv_clostest(m,information_mat,con_store)

        state = information_mat[-1,:,:len(x0)]
        state_end = state[-1,:]
        u = information_mat[-1,-1,len(x0):len(x0)+len(bounds)]
        new_state,t_new = RK4_sub(model,0,0.4,state_end,100,np.array([u]))
        new_state = new_state[-1,:]
        state = np.append(state,[new_state],axis=0)

        end_reward[overall_it,iterations] = state[-1,2] * state[-1,4]

        if iterations == 0 :

            fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))
            ctrl_data = information_mat[-1,:,5:7]
            x_plot,t_plot = RK4_sub_plot(model,0,4,x0,10,ctrl_data)
            old_ctrl_data = np.copy(ctrl_data)
            old_x_plot = np.copy(x_plot)
            old_t_plot = np.copy(t_plot)
            fig.tight_layout()
            plt.subplots_adjust(left=0.1,bottom=0.08,right=0.98,top=0.94,wspace=0.14,hspace=0.33)

            s = np.zeros((6,its,len(x_plot[:,0])))
            sc = np.zeros((2,its,len(ctrl_data[:,0])))
            sp = np.zeros((1,its,len(x_plot[:,0])))


            for i in range(len(x0)):
                axs[i].plot(t_plot,x_plot[:,i],c='k',linewidth=1.4,label=legend[i])
                axs[i].set_ylabel(legend[i])
                s[i,iterations,:] = np.array([x_plot[:,i]])
            sc[0,iterations,:] = ctrl_data[:,0]
            sc[1,iterations,:] = ctrl_data[:,1]

            for i in range(len(axs)):
                axs[i].grid(which='both',alpha=0.5)
            axs[7].plot(t_plot,x_plot[:,2]*x_plot[:,4],c='k',linewidth=1.4)
            sp[0,iterations,:] = np.array([x_plot[:,2]*x_plot[:,4]])

            axs[7].set_ylabel('Production $C_c$ (kmol)')
            axs[-1].set_xlabel('t ($hr$)')
            axs[3].plot(t_plot,[420 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
            axs[4].plot(t_plot,[800 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
            axs[5].step(t[:-1],ctrl_data[:,0],where='post',color='k')
            axs[6].step(t[:-1],ctrl_data[:,1],where='post',color='k')
            axs[5].hlines(ctrl_data[-1,0],t[-2],t[-1],color='k')
            axs[6].hlines(ctrl_data[-1,1],t[-2],t[-1],color='k')


            axs[5].set_ylabel(legend[5])
            axs[6].set_ylabel(legend[6])
            axs[0].set_ylim([0,3.5])
            axs[1].set_ylim([0,3])
            axs[2].set_ylim([0,3])
            axs[3].set_ylim([200,600])
            axs[4].set_ylim([0,1000])
            axs[5].set_ylim([-20,300])
            axs[6].set_ylim([260,520])
            axs[7].set_ylim([0,1500])
            fig.suptitle('Iteration: '+str(iterations))
            plt.savefig(str(iterations)+'.png')
            plt.close()
        else:

            fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))
            ctrl_data = information_mat[-1,:,5:7]
            x_plot,t_plot = RK4_sub_plot(model,0,4,x0,10,ctrl_data)
            fig.tight_layout()
            plt.subplots_adjust(left=0.1,bottom=0.08,right=0.98,top=0.94,wspace=0.14,hspace=0.33)
            for i in range(len(x0)):
                axs[i].plot(t_plot,x_plot[:,i],c='k',linewidth=1.4,label=legend[i])
                axs[i].set_ylabel(legend[i])
                s[i,iterations,:] = np.array([x_plot[:,i]])
            sc[0,iterations,:] = ctrl_data[:,0]
            sc[1,iterations,:] = ctrl_data[:,1]

            for i in range(len(axs)):
                axs[i].grid(which='both',alpha=0.5)

            axs[7].plot(t_plot,x_plot[:,2]*x_plot[:,4],c='k',linewidth=1.4)
            sp[0,iterations,:] = np.array([x_plot[:,2]*x_plot[:,4]])

            axs[7].set_ylabel('Production $C_c$ (kmol)')
            axs[-1].set_xlabel('t ($hr$)')
            axs[3].plot(t_plot,[420 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
            axs[4].plot(t_plot,[800 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
            axs[5].step(t[:-1],ctrl_data[:,0],where='post',color='k')
            axs[6].step(t[:-1],ctrl_data[:,1],where='post',color='k')
            axs[5].hlines(ctrl_data[-1,0],t[-2],t[-1],color='k')
            axs[6].hlines(ctrl_data[-1,1],t[-2],t[-1],color='k')

            axs[5].set_ylabel(legend[5])
            axs[6].set_ylabel(legend[6])
            axs[0].set_ylim([0,3.5])
            axs[1].set_ylim([0,3])
            axs[2].set_ylim([0,3])
            axs[3].set_ylim([200,600])
            axs[4].set_ylim([0,1000])
            axs[5].set_ylim([-20,300])
            axs[6].set_ylim([260,520])
            axs[7].set_ylim([0,1500])

            for i in range(len(x0)):
                axs[i].plot(old_t_plot,old_x_plot[:,i],c='k',linewidth=1.4,label=legend[i],alpha=0.2)
            axs[7].plot(old_t_plot,old_x_plot[:,2]*old_x_plot[:,4],c='k',linewidth=1.4,alpha=0.2)
            axs[5].step(t[:-1],old_ctrl_data[:,0],color='k',alpha=0.2)
            axs[6].step(t[:-1],old_ctrl_data[:,1],color='k',alpha=0.2)
            axs[5].hlines(old_ctrl_data[-1,0],t[-2],t[-1],color='k',alpha=0.2)
            axs[6].hlines(old_ctrl_data[-1,1],t[-2],t[-1],color='k',alpha=0.2)

            fig.suptitle('Iteration: '+str(iterations))
            plt.savefig(str(iterations)+'.png')
            plt.close()

            old_ctrl_data = np.copy(ctrl_data)
            old_x_plot = np.copy(x_plot)
            old_t_plot = np.copy(t_plot)

        

    # images = [] # creating image array

    # for filename in range(its): # iterating over images
        
    #     images.append(imageio.imread(str(filename)+'.png')) # adding each image to the array 
    #     # note see how this follows the standard naming convention
    #     #os.remove(str(filename)+'.png') # this then deletes the image file from the folder

    # imageio.mimsave('reactor_dynamics_course_'+str(overall_it)+'.gif', images) # this then saves the array of images as a gif

# bayes = False

# information_mat_add,con_vec = q_controller(m,m_con,1,information_mat)
# con_vec = np.reshape(con_vec,(len(con_vec),1))

# information_mat = np.append(information_mat,[information_mat_add],axis=0)
# con_store = np.append(con_store,[con_vec],axis=0)


# m = GP_train(information_mat)
# m_con = GP_train_con(information_mat,con_store)


# state = information_mat[-1,:,:len(x0)]
# state_end = state[-1,:]
# u = information_mat[-1,-1,len(x0):len(x0)+len(bounds)]
# new_state,t_new = RK4_sub(model,0,0.4,state_end,100,np.array([u]))
# new_state = new_state[-1,:]
# state = np.append(state,[new_state],axis=0)

# end_reward[overall_it,iterations] = state[-1,2]


s = s[:,20:,:]
sc = sc[:,20:,:]
sp = sp[:,20:,:]


av_states = np.mean(s,axis=1)
av_ctrl = np.mean(sc,axis=1)
std_states = np.std(s,axis=1)
std_ctrl = np.std(sc,axis=1)

av_prod = np.mean(sp,axis=1)
std_prod = np.std(sp,axis=1)

fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))

for i in range(len(x0)):
    axs[i].plot(t_plot,av_states[i,:],c='k',linewidth=1.4,label=legend[i])
    axs[i].fill_between(t_plot,av_states[i,:]-std_states[i,:],av_states[i,:]+std_states[i,:],color='k',alpha=0.2)
    axs[i].set_ylabel(legend[i])
for i in range(len(axs)):
                axs[i].grid(which='both',alpha=0.5)
axs[7].plot(t_plot,av_prod[0,:],c='k',linewidth=1.4)
axs[7].fill_between(t_plot,av_prod[0,:]-std_prod[0,:],av_prod[0,:]+std_prod[0,:],color='k',alpha=0.2)

axs[7].set_ylabel('Production $C_c$ (kmol)')
axs[-1].set_xlabel('t ($hr$)')
axs[3].plot(t_plot,[420 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
axs[4].plot(t_plot,[800 for i in range(len(t_plot))],c='r',linewidth=1.4,linestyle='--')
axs[5].step(t[:-1],av_ctrl[0,:],where='post',color='k')
axs[6].step(t[:-1],av_ctrl[1,:],where='post',color='k')
axs[5].hlines(av_ctrl[0,-1],t[-2],t[-1],color='k')
axs[6].hlines(av_ctrl[1,-1],t[-2],t[-1],color='k')


axs[5].fill_between(t[:-1],av_ctrl[0,:]-std_ctrl[0,:],av_ctrl[0,:]+std_ctrl[0,:],step='post',color='k',alpha=0.2)
axs[6].fill_between(t[:-1],av_ctrl[1,:]-std_ctrl[1,:],av_ctrl[1,:]+std_ctrl[1,:],step='post',color='k',alpha=0.2)

axs[5].fill_between(t[-2:],av_ctrl[0,-1]-std_ctrl[0,-1],av_ctrl[0,-1]+std_ctrl[0,-1],step='post',color='k',alpha=0.2)
axs[6].fill_between(t[-2:],av_ctrl[1,-1]-std_ctrl[1,-1],av_ctrl[1,-1]+std_ctrl[1,-1],step='post',color='k',alpha=0.2)



axs[5].set_ylabel(legend[5])
axs[6].set_ylabel(legend[6])
fig.tight_layout()
plt.subplots_adjust(left=0.1,bottom=0.08,right=0.98,top=0.94,wspace=0.14,hspace=0.33)
fig.suptitle('Deployment Run')
plt.savefig('deployment_run.png')
 




