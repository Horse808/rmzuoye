import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('stock_prices.csv')
days=df['Day'].values
prices=df['Price'].values
F=np.array([[1,1],[0,1]])
H=np.array([[1,0]])
Q=np.array([[0.5,0],[0,0.1]])
R=np.array([[5.0]])
x=np.array([[prices[0]],[0]])
P=np.array([[10,0],[0,10]])
filtered_prices=[]
estimated_trends=[]
predictioms=[]
for i,price in enumerate(prices):
    x_pred=F@x
    P_pred=F@P@F.T+Q
    y=price-(H@x_pred)[0,0]
    S=H@P_pred@H.T+R
    K=P_pred@H.T/S[0,0]
    P=(np.eye(2)-K@H)@P_pred
    x=x_pred+K*y
    filtered_prices.append(x[0,0])
    estimated_trends.append(x[1,0])
    x_next=F@x
    predictioms.append(x_next[0,0])
filtered_prices=np.array(filtered_prices)
estimated_trends=np.array(estimated_trends)
predictioms=np.array(predictioms)
def predict_future(future_days=10):
    x_future=x.copy()
    future=[]
    for _ in range(future_days):
        x_future=F@x_future
        future.append(x_future[0,0])
    return future
fig,axes=plt.subplots(2,1,figsize=(14,12))
axes[0].plot(days,prices,'b.',alpha=0.5,markersize=3,label='yuanshi')
axes[0].plot(days,filtered_prices,'r-',linewidth=1.5,label='kalman')
axes[0].set_xlabel('day')
axes[0].set_ylabel('price')
axes[0].set_title('kalman')
axes[0].legend()
future_days=np.arange(len(prices),len(prices)+10)
future_10=predict_future(10)
axes[1].plot(days[-30:],prices[-30:],'b-',linewidth=1.5,label='lishijiage')
axes[1].plot(future_days,future_10,'r--',linewidth=1.5,marker='o',markersize=4,label='weilaiyuce')
axes[1].axvline(x=days[-1],color='gray',linestyle='--',alpha=0.5)
axes[1].set_xlabel('day')
axes[1].set_ylabel('price')
axes[1].set_title('future 10 days')
axes[1].legend()
axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.show()