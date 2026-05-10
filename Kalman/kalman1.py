import numpy as np
import matplotlib.pyplot as plt
file_names=["homework_data_1.txt","homework_data_2.txt","homework_data_3.txt","homework_data_4.txt"]
def kalman_filter(data):
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
    return np.array(filtered)
t_data=[]
z_data=[]
filtered_data=[]
poly_data=[]
fig,axes=plt.subplots(2,2,figsize=(14,10))
titles=['1','2','3','4']
for file_name in file_names:
    data=np.loadtxt(file_name)
    filtered=kalman_filter(data)
    t=data[:,0]
    z=data[:,1]
    degree=5
    coeffs=np.polyfit(t,filtered,degree)
    poly=np.poly1d(coeffs)
    t_data.append(t)
    z_data.append(z)
    filtered_data.append(filtered)
    poly_data.append(poly)
    print(poly)
for i,ax in enumerate(axes.flat):
    t=t_data[i]
    z=z_data[i]
    filtered=filtered_data[i]
    poly=poly_data[i]
    ax.plot(t,z,'b.',alpha=0.3,label='yuanshidata')
    ax.plot(t,filtered,'r-',linewidth=1.5,label='kalman')
    ax.plot(t,poly(t),'g--',linewidth=1.5,label=f'nihe(degree={degree})')
    ax.set_xlabel('time')
    ax.set_ylabel('zhi')
    ax.set_title(f'{titles[i]}-kalman')
    ax.legend(loc='upper left',fontsize=9)
    ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.suptitle('homework1',fontsize=14,y=1.02)
plt.show()