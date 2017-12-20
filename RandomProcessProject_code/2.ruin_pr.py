from random import gauss
import numpy as np
import matplotlib.pyplot as plt

## =================================================================
##  基本设置
##  调节u 和 T
## =================================================================
u = 20
c1 = 5
c2 = 1.3/20
λ = 1
T = 50
H = 0.75
amp = 20/20

实验次数 = 100
破产次数 =0
plt.axhline(y=0.0, color='g', linestyle='-')



## =================================================================
##  理论上界
## =================================================================

from scipy.stats import norm
from math import exp
sigma=0.55
phi=norm.cdf(u*T**(-H) +c2*T**(1-H))
phi = norm.cdf((u+c2*T)/(sigma*T**H))
print("u=%.4f  H=%.4f"%(u,H))
print("theoretical upper bound = %.3f"%(1-phi+exp(-2*u*c2*T**(1-2*H)/sigma**2)*phi))



## =================================================================
##  分形布朗运动模拟
## =================================================================


from fbm import FBM
from tqdm import trange

f = FBM(n=1000, hurst=H, length=T, method='daviesharte')
for i in trange(实验次数,ncols=80):
    fbm_ts = f.times()
    fbm_asset = f.fbm()
    fbm_asset=[u+c2*fbm_ts[i] -amp**H*fbm_asset[i] for i in range(len(fbm_asset))]

    for i in range(len(fbm_asset)):
        if fbm_asset[i]<0:
            fbm_ts =fbm_ts[:i+1]
            fbm_asset =fbm_asset[:i+1]
            break

    if fbm_asset[-1:][0]<0:
        #print("Ruin time=",fbm_ts[len(fbm_asset)-1],"asset=",fbm_asset[-1:][0])
        破产次数+=1



print("Estimated Ruin Probability=%.3f"%(破产次数/实验次数))
#plt.plot(fbm_ts,fbm_asset,label="FractionBrownion")
#plt.xlabel("Time")
#plt.ylabel("Asset")
#plt.legend()
#plt.show()