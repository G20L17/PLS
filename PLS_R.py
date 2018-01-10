import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import scale, maxabs_scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
import sklearn.metrics as sklm
from datetime import datetime

print("%s", str(datetime.now()))
RANDOM_SEED = 193
Nsample=100000
Nsnr=100
path='/home/machinelearningstation/PycharmProjects/multiclassification'
df=pd.read_csv(path+'/data/spec_C2H6_all_9_pxl_1000_samples_'+str(Nsample)+'_SNR_'
               +str(Nsnr)+'_dB.csv')

target = df[df.columns[-9:]]
target.plot(kind='box')

plt.savefig('data_'+str(Nsample)+'_'+str(Nsnr)+'db')
#plt.show()
data = df.drop(df.columns[-9:], axis=1)

print(data.shape)

X=scale(pd.DataFrame.as_matrix(data))
Y=pd.DataFrame.as_matrix(target)

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)
M=30
kf = KFold(n_splits=5, random_state=RANDOM_SEED)


mse=[]
for i in np.arange(1, M+1):
    pls = PLSRegression(n_components=i)
    score = cross_val_score(pls, X_train, Y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    print ('component= %d, MSE= %e' %(i, -score))
    mse.append(-score)
np.savetxt('MSE_'+str(Nsample)+'_'+str(Nsnr)+'db', mse, delimiter=',')
plt.plot(np.arange(1, M+1), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')

plt.savefig('com_'+str(Nsample)+'_'+str(Nsnr)+'db')
#plt.show()
ind=0         #np.argmin(np.array(mse))
n_component=np.arange(1, M+1)[ind]
pls2=PLSRegression(n_components=n_component)
pls2.fit(X_train, Y_train)
y_true=Y_test
y_predict=pls2.predict(X_test)
mse_test=sklm.mean_squared_error(y_true, y_predict)
print('component= %d, MSE= %e' %(n_component, mse_test))
np.savetxt('_TestY_' + str(Nsample) + '_' + str(Nsnr) + 'db', y_predict, delimiter=',')
np.savetxt('_PredY_' + str(Nsample) + '_' + str(Nsnr) + 'db', y_true, delimiter=',')



print("%s", str(datetime.now()))



'''''
x=np.arange(1, M+1)
y=np.array(mse)
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_min(x,y, ax=None):
    xmax = x[np.argmin(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

annot_min(x,y)

plt.xlabel('Number of principal components in regression')
plt.ylabel('Mean Square Error')
plt.savefig('/home/machinelearningstation/PycharmProjects/spectrum2/N_component.png')
'''