

import numpy as np
import matplotlib.pyplot as plt
from smt.applications.mfk import MFK, NestedLHS
import itertools
import pandas
import seaborn as sns
# Problem set up
xlimits = np.array([[0.50,1.5],[0.42,1.58]])


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy.stats import norm, gaussian_kde
xt_c = np.array([[0.5,0.42],[0.50,0.65],[0.50,1],[0.5,1.35],[0.5,1.58],[0.7,0.42],[0.7,0.65],[0.7,1],[0.7,1.35],[0.7,1.58],[1,0.42],[1,0.65],[1,1],[1,1.35],[1,1.58],\
[1.3,0.42],[1.3,0.65],[1.3,1],[1.3,1.35],[1.3,1.58],[1.5,0.42],[1.5,0.65],[1.5,1],[1.5,1.35],[1.5,1.58]])# CREATE POINT SET FOR EXPENSIVE MODEL.
#xt_e=np.array([[0.5,0.42],[0.5,1],[0.5,1.58],[1,0.42],[1,1],[1,1.58],[1.5,0.42],[1.5,1],[1.5,1.58]])# CREATE POINT SET FOR CHEAP MODEL.
xt_e=np.array([[0.5,0.42],[0.5,1],[0.5,1.58],[1,0.42],[1,1],[1,1.58],[1.5,0.42],[1.5,1],[1.5,1.58]])# CREATE POINT SET FOR CHEAP MODEL.



yt_c=np.array([0.4986808747,
0.475976665,
0.4777031069,
0.4923418646,
0.5028265703,
0.4753060066,
0.4595472371,
0.4620449908,
0.4691087625,
0.4742488188,
0.4513367251,
0.4396361379,
0.4368311097,
0.437009123,
0.4393375827,
0.4180588293,
0.4084524096,
0.404768885,
0.4059021486,
0.4059021486,
0.3861760934,
0.3767324971,
0.3724887529,
0.3747228606,
0.3767415899])
yt_e=np.array([0.5047597937,
0.5222562742,
0.5985461618,
0.4187630715,
0.3821620197,
0.4202147777,
0.2783293064,
0.3080605842,
0.3080605842])
sm = MFK(theta0=xt_e.shape[1] * [1.0])


# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(xt_c, yt_c, name=0)
# high-fidelity dataset without name
sm.set_training_values(xt_e, yt_e)

# train the model
sm.train()



testx = np.linspace(0.5,1.5, 101, endpoint=True)

testy= np.linspace(0.42,1.58, 101, endpoint=True)


x = [(a, b) for a in testx for b in testy] 
x=np.array([x])
x=np.reshape(x,[10201,2])

# query the outputs
y = sm.predict_values(x)
print(len(y))
mse = sm.predict_variances(x)
derivs = sm.predict_derivatives(x, kx=0)

print(mse)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
YP=np.reshape(y,[101,101])
'''
# Plot the surface.
surf = ax.plot_surface( xt_c[:,0] , xt_c[:,1] , yt_c, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0.2,0.6)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')


'''

np.savetxt('testcokrigall.txt', y, delimiter='\n')

import seaborn as sns
data = pandas.read_csv('dns9data.csv')
np.savetxt('testvariable.txt', data, delimiter='\n')
  
# draw jointplot with
# scatter kind
sns.jointplot(x="alpha", y="gamma", kind = "kde", data = data)
# show the plot
plt.show()








