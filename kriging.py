import numpy as np
import matplotlib.pyplot as plt

from smt.surrogate_models import KRG

xt=np.array([[0.5,0.42],[0.5,1],[0.5,1.58],[1,0.42],[1,1],[1,1.58],[1.5,0.42],[1.5,1],[1.5,1.58]])# CREATE POINT SET FOR CHEAP MODEL.
yt=np.array([0.5047597937,
0.5222562742,
0.5985461618,
0.4187630715,
0.3821620197,
0.4202147777,
0.2783293064,
0.3080605842,
0.3080605842])

sm = KRG(theta0=xt.shape[1] * [1.0])
sm.set_training_values(xt, yt)
sm.train()


testx = np.linspace(0.5,1.5, 101, endpoint=True)

testy= np.linspace(0.42,1.58, 101, endpoint=True)


x = [(a, b) for a in testx for b in testy] 
x=np.array([x])
x=np.reshape(x,[10201,2])
y = sm.predict_values(x)
# estimated variance
s2 = sm.predict_variances(x)
# derivative according to the first variable
dydx = sm.predict_derivatives(xt, 0)
fig, axs = plt.subplots(1)
'''
# add a plot with variance
axs.plot(xt, yt, "o")
axs.plot(x, y)
axs.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(s2)),
    np.ravel(y + 3 * np.sqrt(s2)),
    color="lightgrey",
)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction", "Confidence Interval 99%"],
    loc="lower right",
)
'''
np.savetxt('x1.txt', x[:,0], delimiter='\n')
np.savetxt('x2.txt', x[:,1], delimiter='\n')
np.savetxt('testkrigdns.txt', y, delimiter='\n')
plt.show()
