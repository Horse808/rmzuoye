import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
data=np.loadtxt("homework_data_1.txt")
t=data[:,0]
z=data[:,1]
dt=t[1]-t[0]
F=np.array([[1,dt],[0,1]])
H=np.array([[1,0]])
Q=np.array([[0.01,0],[0,0.01]])
R=np.array([[0.5]])
x=np.array([[z[0]],[0.0]])
P=np.array([[1,0],[0,1]])
filtered=[]
for measurement in z:
    x=F@x
    P=F@P@F.T+Q
    y=np.array([[measurement-(H@x)[0,0]]])
    S=H@P@H.T+R
    K=P@H.T@np.linalg.inv(S)
    x=x+K@y
    P=(np.eye(2)-K@H)@P
    filtered.append(x[0,0])
filtered=np.array(filtered)
def poly_func(t,*coeffs):
    return sum(c*(t**i)for i,c in enumerate(coeffs))
degree=5
coeffs=np.polyfit(t,filtered,degree)
poly=np.poly1d(coeffs)
print("fangcheng:")
print(poly)
plt.figure(figsize=(12,6))
plt.plot(t,z,'b.',alpha=0.3,label='yuanshidata')
plt.plot(t,filtered,'r-',label='kalman')
plt.plot(t,poly(t),'g--',label=f'nihe(degree={degree})')
plt.xlabel('time')
plt.ylabel('zhi')
plt.legend()
plt.title('homework1')
plt.grid(True)
plt.show()