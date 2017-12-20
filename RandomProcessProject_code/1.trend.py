from random import gauss
import numpy as np
import matplotlib.pyplot as plt

## =================================================================
##  基本设置
## =================================================================
u = 30 
c1 = 5
c2 = 1.3
λ = 1
T = 50
H = 0.75
amp = 20

plt.axhline(y=0.0, color='g', linestyle='-')

## =================================================================
##  平稳过程仿真
## =================================================================
def stationary(mu=6.45, sigma=5.0):
    return gauss(mu,sigma)


## =================================================================
##  泊松过程仿真
## =================================================================


# generate a canonical (mean=1) list of exponential random variables
def randExp(N = 1):
    return -np.log(np.random.uniform(0.,1.,N))

# generate a list of arrival times for the Poisson process with integrated rate function inverse: irfi
def poissonProcess(irfi, T):
    t = 0
    ts = []
    yc = 0
    while (t<T):
        yc += randExp()
        t = irfi(yc)
        ts.extend(t)
    return ts[:-1]


# the integrated rate function inverse corresponding to f(t) = λ
def exampleHomogenousIRFI(y):
    t = y/λ
    return t

def plotPoissonProcess(pp, T):
    tt = np.linspace(0,T,1001)
    pps = 0*tt
    for t in pp:
        pps[tt>t] += 1
    plt.plot(tt,pps)
    plt.xlabel('time axis')
    plt.ylabel('event counter')
    plt.show()

def getPoissonProcess(pp, T):
    tt = np.linspace(0,T,1000)
    pps = 0*tt
    for t in pp:
        pps[tt>t] += 1
    
    return (tt,pps)


## =================================================================
##  生成0到T时刻的泊松过程计数，每次计数加上一个白噪声事件。
## =================================================================
pph = poissonProcess(exampleHomogenousIRFI, T)
#plotPoissonProcess(pph, T)
ts, pCount=  getPoissonProcess(pph,T)
#print(ts)
#print(pCount)


asset = u
simuAsset = []
simuAsset.append(asset)
for i in range(1, 1+len(ts)-1):
    asset += c1*(ts[i]-ts[i-1])
    lossTimes = int(pCount[i]-pCount[i-1])
    for i in range(lossTimes):
        asset -= stationary()
    simuAsset.append(asset)
    if asset<0:
        break

plt.plot(ts[:len(simuAsset)], simuAsset,label="Poisson")



print("Ruin time=",ts[len(simuAsset)-1],"asset=",(simuAsset[-1:]))

## =================================================================
##  分形布朗运动模拟
## =================================================================
from fbm import FBM
f = FBM(n=1000, hurst=H, length=T, method='daviesharte')
fbm_ts = f.times()
fbm_asset = f.fbm()
fbm_asset=[u+c2*fbm_ts[i] -amp**H*fbm_asset[i] for i in range(len(fbm_asset))]

for i in range(len(fbm_asset)):
    if fbm_asset[i]<0:
        fbm_ts =fbm_ts[:i+1]
        fbm_asset =fbm_asset[:i+1]
        break

print("Ruin time=",fbm_ts[len(fbm_asset)-1],"asset=",(fbm_asset[-1:]))

plt.plot(fbm_ts,fbm_asset,label="FractionBrownion")
plt.xlabel("Time")
plt.ylabel("Asset")
plt.legend()
plt.show()