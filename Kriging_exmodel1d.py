########1D Simple Kriging for y=(6*x-2)^2*sin(12*x-4)
########Shaima Magdaline Dsouza,30 Jun 2022, smdsouza@kth.se
########Matern model with nu=3/2 is used for the variogram
########References: 
########Following code was modified
########https://sourceforge.net/p/geoms2/wiki/Kriging/
########Lectures from GeostatsGuy Prof.Michael Pyrcz, University of Texas at Austin 
######## https://github.com/GeostatsGuy/ExcelNumericalDemos/blob/master/Simple_Kriging_Demo.xlsx

from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
def SK(x,v,variogram):
    
    cov_distance= np.zeros((x.shape[0],x.shape[0]))
    K = np.zeros((x.shape[0],x.shape[0]))
    
    for i in range(x.shape[0]-1):
       
        cov_distance[i,i:]=np.sqrt((x[i:]-x[i])**2)
        
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if cov_distance[i,j]!=0:
                amp=(variogram[0]*(1+np.sqrt(6)*cov_distance[i,j]/variogram[1])*np.exp(-np.sqrt(6)*cov_distance[i,j]/variogram[1]))
                K[i,j]=variogram[0]*1-amp
    K = K + K.T
    return K
  
def Krig_meta(x,prob,v,variogram,K):
    for i in range(probe.shape[0]):
             
             W= np.zeros((v.shape[0],1))
             distance= np.zeros((v.shape[0],1))
             for j in range(x.shape[0]):
                distance[j] = np.sqrt((prob[i]-x[j])**2)
             
             amplitudes =(variogram[0]*(1+np.sqrt(6)*distance[:]/variogram[1])*np.exp(-np.sqrt(6)*distance[:]/variogram[1]))
             M =np.ones((4, 1))*variogram[0]-(amplitudes[:])
             W = LA.solve(K,M)
            
             
             grid[i] = np.dot(W.T,(v[:]-v[:].mean()))+v[:].mean()
         
             
    return grid

np.random.seed(123433789) # GIVING A SEED NUMBER FOR THE EXPERIENCE TO BE REPRODUCIBLE
probe =  np.linspace(0, 1, 100)# float32 gives us a lot precision

def func(x):
    return (6*x-2)**2*(np.sin(12*x-4))
    
x = np.array([0,0.4,0.6,1.0])# CREATE POINT SET.

v= func(x)
exacty=func(probe)
grid = np.zeros((100,1),dtype='float32') 

CM = SK(x,v,(5,0.25))

res = Krig_meta(x,probe,v,(5,0.25),CM)



plt.scatter(x,v,c="black")
plt.plot(probe,res,c= "red", marker='.')
plt.plot(probe,exacty,c= "blue", marker='.')
plt.legend(['Kriging','Exact', 'Data'])
plt.title('y=(6*x-2)^2*sin(12*x-4)')
plt.gca()
plt.show()

