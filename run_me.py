import numpy as np
import K_UCBVI_lib
def plot_func(Data_save,color_line,plt,Final,Step,plt_label,y_label):
        import math
        dd  = Data_save
        if (plt == 0):
                        import matplotlib.pyplot as plt
                       
        m  = np.mean(dd,axis = 0)
        sd = np.std(dd,axis =0)/(math.sqrt(dd.shape[0]))
        mup = m+sd
        mlp = m-sd
        m = m
        mup = mup
        
        mlp = mlp
        
       
        color_area = color_line + (1-color_line)*2.3/4 
        
        plt.plot(range(Step,Final,Step),m,color= color_line,label = plt_label)
        plt.fill_between(range(Step,Final,Step),mup,mlp,color=color_area)
        plt.xlabel('Number of Training Steps')
        plt.ylabel(y_label)
    
        plt.grid(True)
        plt.legend()
        return  plt

import UCBVI_lib
Train_steps = 500

Interval = 20

Repititions = 20

Est_err,CumuReg = UCBVI_lib.UCBVI(Train_steps,Interval,Repititions)

color_line = np.array([179, 63, 64])/255 

plt_label= 'UCBVI on simple grid world'

plot_func(Est_err,color_line,0,Train_steps,Interval,plt_label,'Estimation_Err')

plot_func(CumuReg,color_line,0,Train_steps,Interval,plt_label,'Cumulative Regret')



import UCBVI_lib
Train_steps = 500

Interval = 20

Repititions = 3

K  = 2
Est_err,CumuReg = K_UCBVI_lib.K_UCBVI(Train_steps,Interval,Repititions,K)

color_line = np.array([179, 63, 64])/255

plt_label= 'K-UCBVI on modified grid world (K=2)'

plt  = plot_func(CumuReg,color_line,0,Train_steps,Interval,plt_label,'Cumulative Regret')
plt1  = plot_func(Est_err,color_line,0,Train_steps,Interval,plt_label,'Estimation_Err')


K  = 4
Est_err,CumuReg = K_UCBVI_lib.K_UCBVI(Train_steps,Interval,Repititions,K)

plt_label= 'K-UCBVI on modified grid world (K=4)'

plt  = plot_func(CumuReg,color_line,0,Train_steps,Interval,plt_label,'Cumulative Regret')
plt1  = plot_func(Est_err,color_line,plt1,Train_steps,Interval,plt_label,'Estimation_Err')

K  = 8
Est_err,CumuReg = K_UCBVI_lib.K_UCBVI(Train_steps,Interval,Repititions,K)

plt_label= 'K-UCBVI on modified grid world (K=8)'
plt1  = plot_func(Est_err,color_line,plt1,Train_steps,Interval,plt_label,'Estimation_Err')
plt  = plot_func(CumuReg,color_line,0,Train_steps,Interval,plt_label,'Cumulative Regret')

plt.show()
plt1.show()





