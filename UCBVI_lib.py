import numpy as np
import random
H = 6
state_space = []
policy = [0,0,3,0,0,2,3,0,0,0,3,0,3,3,1,0]
policy_opt =[3, 3, 3, 0, 1, 1, 3, 0, 0, 0, 3, 0, 3, 3, 3, 0] 
gamma = 0.95
error = 0.01
# action up->0,down->1,left->2,right->3



for i in range(1,5):
   for j in range(1,5):
            state_space.append((i,j))

pi_opt = np.zeros((16,4))
for s in state_space:
    pi_opt[state_space.index(s),policy_opt[state_space.index(s)]] = 1
   

def translatepolicy(P):
   Trueactions =['up','down','left','right']
   TP =[]
   for i in P:
           TP.append(Trueactions[i])
   return TP    

def reward(s):
    r  = 0
    if (s == (3,2)):
                     r = -1;
    if (s == (4,4)):
                     r = 1;    
    return r 

def reward_vector(s):
    reward_vector = np.zeros(len(s))
    for i in s: 
       reward_vector[s.index(i)] = reward(i)
    return reward_vector 

def takeaction(s,a,state_space):
    snew = s
    if (a==0):
                snew =  (snew[0] + 1,snew[1])   
    if (a==1):
                snew =  (snew[0] - 1,snew[1])    
    
    if (a==2):
                snew =  (snew[0] ,snew[1]-1) 
 
    if (a==3):
                snew =  (snew[0] ,snew[1]+1) 

    if (snew not in state_space or snew == (2,3) or s == (3,2) or s==(4,4)):
                                                 snew = s;
    return snew


def Transition_P(policy,state_space):
    P = np.zeros((len(state_space),len(state_space)))
    acts = [0,1,2,3]
    for s in state_space:
        s_index = state_space.index(s)
        for a in acts:
           pr = 0.03
           
           if(a==policy[s_index]):
                                   pr = 0.91  
               
           snew = takeaction(s,a,state_space) 
           P[s_index,state_space.index(snew)] =  P[s_index,state_space.index(snew)] + pr   
  
    return P


def Transition_P_givenstate(s,action,state_space):
        P = np.zeros(len(state_space))
        acts = [0,1,2,3]
    
        s_index = state_space.index(s)
        for a in acts:
           pr = 0.03
        
           if(a==action):
                                   pr = 0.91   
               
           snew = takeaction(s,a,state_space) 
           P[state_space.index(snew)] =  P[state_space.index(snew)] + pr   
  
        return P


def sample_episode(s0,policy,state_space,N,P,Np):
    import numpy as np
    state_vec = []

    state_vec.append(s0)
    steps = 0
    while(s0 != (2,3) and s0 != (3,2) and s0!=(4,4) ):
        if (steps >=  H):
                         break 
        action =  np.random.choice([0 ,1, 2 ,3], size=1,  p=policy[state_space.index(s0),:])      
        N[state_space.index(s0),action] = N[state_space.index(s0),action] + 1
        spr = s0
        Pr =  Transition_P_givenstate(s0,action,state_space)
            
        s0 =  state_space[int(np.random.choice(list(range(16)),1,p=Pr))]
        Np[state_space.index(s0),state_space.index(spr),action] = Np[state_space.index(s0),state_space.index(spr),action]  + 1
        for s1 in state_space:
            P[state_space.index(s1),state_space.index(spr),action] =  Np[state_space.index(s1),state_space.index(spr),action]/ N[state_space.index(spr),action]
            
        state_vec.append(s0)
        steps = steps + 1

    while(1):
      if (steps >= H):  
                      break
      state_vec.append(s0) 
      steps = steps + 1  
    return state_vec,N,P,Np




def sample_episode22(s0,policy,state_space):
    import numpy as np
    state_vec = []

    state_vec.append(s0)
    steps = 0
    while(s0 != (2,3) and s0 != (3,2) and s0!=(4,4) ):
        if (steps >=  H):
                         break 
        action =  np.random.choice([0 ,1, 2 ,3], size=1,  p=policy[state_space.index(s0),:])      
       
        spr = s0
        Pr =  Transition_P_givenstate(s0,action,state_space)
            
        s0 =  state_space[int(np.random.choice(list(range(16)),1,p=Pr))]
       
            
        state_vec.append(s0)
        steps = steps + 1

    while(1):
      if (steps >= H):  
                      break
      state_vec.append(s0) 
      steps = steps + 1  
    return state_vec

def sample_trajectory(s0,policy,state_space,Pr):
    import numpy as np
    state_vec = []
    action_vec =[]
    p_tr = 1
    state_vec.append(s0)
    steps = 0
    while(s0 != (2,3) and s0 != (3,2) and s0!=(4,4) ):
        if (steps >=  H):
                         break 
        action =  np.random.choice([0 ,1, 2 ,3], size=1,  p=policy[state_space.index(s0),:])
        action_vec.append(action[0])
        p_tr = p_tr *  policy[state_space.index(s0),action]
        
        spr = s0
        Prb =  Pr[:,state_space.index(s0),action]
       
        s0 =  state_space[int(np.random.choice(list(range(16)),1,p=  Prb[:,0]/(np.sum(Prb[:,0]))))]

        p_tr = p_tr *  Prb[state_space.index(s0),0]/(np.sum(Prb[:,0]))
       
            
        state_vec.append(s0)
        steps = steps + 1

  
    return state_vec,action_vec,p_tr




def approximate_return(T,w,sigma,t):
    import math
    N = H 
    final_state = T[-1];
    dist = (4-final_state[1]) + (4-final_state[0]) 
    phie = -dist/3+1
    e = math.e
    mu_hat = 1/(1+e**(-w*phie)) 
    rho = math.log(3+4*t)+2*math.log(N/0.95)+0.5
    beta = 1+3+rho*((1+3)**(0.5)+rho)**(1.5)
    return min(mu_hat[0],1)
    


def generate_initial_policy(state_space):
    pi = np.zeros((len(state_space),4))
    for s in state_space:
         pi[state_space.index(s):] = np.array([0.25,0.25,0.25,0.25])
    return pi     


def apply_phie(T):
    import math
    final_state = T[-1];
    dist = (4-final_state[1]) + (4-final_state[0]) 
    phie = -dist/3+1
   
    return phie

def theta_to_policy(theta):
    from sklearn.preprocessing import normalize
    theta = theta.reshape(16,4)
    pi = np.exp(theta)
    pi  = normalize(pi, axis=1, norm='l1')
    return pi

def calculate_reward(T):
    import math
    final_state = T[-1];
    dist = (4-final_state[1]) + (4-final_state[0]) 
    phie = -dist/3+1
    e = math.e
    mu = 1/(1+e**(-3*phie))

    y = np.random.binomial(1, mu, size=1)
    return y[0]

def grad_cal(Act,Trj,Pi,state_space):
    GD = np.zeros((16,4))
    t = 0
    for a in Act:
        s = Trj[t]
        
        s_ind = state_space.index(s)
        
        G = np.zeros((16,4))
        G[s_ind,:] = -1*Pi[s_ind,:]
        G[s_ind,a] = 1+ G[s_ind,a]
        GD = GD + G
        t = t + 1
    return GD.reshape(64)

import numpy as np
from scipy.optimize import minimize

# Define a function with additional constants
def objective_function(x, f, y):
   tmp = 1/(1+np.exp(-1*x*f))
   return -1*np.dot(y,np.log(tmp)) -1*np.dot(1-y,np.log(1-tmp)) + x**2/2
    
def solve_optim(feedback,feature_vec):
    # Initial guess
    initial_guess = [1]
    
    # Additional constants
    constant1 = np.array(feature_vec)
    constant2 = np.array(feedback)
    
    #Call the minimize function with args
    result = minimize(objective_function, initial_guess, args=(constant1, constant2), method='BFGS')
    
    # Print the result
    return result.x

def UCBVI(Final,Step,Rept):
        import numpy as np           
        import math          
        final_out_Reward =[]
        final_out_Err = []
       
        N = np.zeros((16,4))
        Np = np.zeros((16,16,4))
        P = np.zeros((16,16,4))+1/16
        Trajectories = []
        feedback =[]
        feature_vec =[] 
        tmp_out_Reward = []
        tmp_out_Err = []
        Pi =  generate_initial_policy(state_space)
        for enddd in range(Step,Final,Step):
            tmp_out_Reward = []
            tmp_out_Err = []
            for numm in range(Rept):
                    N = np.zeros((16,4))
                    Np = np.zeros((16,16,4))
                    P = np.zeros((16,16,4))+1/16
                    Trajectories = []
                    feedback =[]
                    feature_vec =[] 
                    if 'theta'  in locals():
                                       del(theta)
                    Pi =  generate_initial_policy(state_space)
    
                    for i in range(enddd):
                        #if (i > 0):
                         #              w_est = solve_optim(feedback,feature_vec)
                            
                        s0 =  state_space[int(np.random.choice(list(range(16)),1,p=None))]
                        T,N,P,Np = sample_episode(s0,Pi,state_space,N,P,Np)
                        Trajectories.append(T) 
                        feedback.append(calculate_reward(T))
                        feature_vec.append(apply_phie(T))
                        sigma = (2.71828)**3+np.dot(feature_vec,feature_vec)
                    
                    w_est = solve_optim(feedback,feature_vec)
                    #print(w_est)
                    
                    
                    if 'theta' not in locals():
                                       theta = np.zeros(64)
                        
                    Returns = np.zeros(10)
                    for ii in range(500):
                            dtheta = np.zeros(64)
                            Tr = 0 
                            for jj in range(300):
                                 
                                    s0 =  state_space[int(np.random.choice(list(range(16)),1,p=None))]
                                    T,A,p = sample_trajectory(s0,Pi,state_space,P)
                                   
                                    #print(A)
                                    #print(p)
                                    rr  = approximate_return(T,w_est,sigma,100)
                                    Tr = Tr + approximate_return(T,np.array([3]),sigma,100)
                                    G = grad_cal(A,T,Pi,state_space)
                                    #print(G)
                                    #print(rr)
                                    dtheta = dtheta +rr*G
                            #Returns[ii] = Tr/300
                            #print(Returns[ii])
                            theta1 = theta + 0.01*dtheta
                            Pi = theta_to_policy(theta1)
                            #print(np.linalg.norm(theta-theta1, ord=1))  
                            #print(ii)
                            theta = theta1
    
    
                    Tr=0
                    for jj in range(500):
             
                     
                        T = sample_episode22((1,1),Pi,state_space)
                        Tr = Tr + approximate_return(T,np.array([3]),sigma,100)
                       
                
                    tmp_out_Reward.append(0.87-Tr/500)
                    tmp_out_Err.append(abs(3-w_est))
                    #print(enddd,numm,tmp_out_Reward)      
                     
            final_out_Reward.append(tmp_out_Reward)
            final_out_Err.append(tmp_out_Err)
        Reward  = np.array(final_out_Reward) 
        CumuReward  =  np.cumsum(np.transpose(Reward),1)

        Err  = np.array(final_out_Err) 
        Err  =  np.transpose(Err)
        return Err[0],CumuReward
  
            