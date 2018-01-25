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
Nsnr=50
train_size=2500
path='/home/machinelearningstation/PycharmProjects/multiclassification'
df=pd.read_csv(path+'/data/spec_C2H6_all_9_pxl_1000_samples_'+str(Nsample)+'_SNR_'
               +str(Nsnr)+'_dB.csv')
df=df.sample(frac=1)
target = (df[df.columns[-9:]].apply(lambda x: x!=0)). astype(int)
data=df.drop(df.columns[-9:],axis=1)
X=scale(pd.DataFrame.as_matrix(data))
Y=pd.DataFrame.as_matrix(target)

X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)
X_train=X_train[0:train_size]
Y_train=Y_train[0:train_size]
Mtrain, Ntrain=Y_train.shape
Mtest, Ntest=Y_test.shape
y_predict=np.zeros((Mtest, Ntest))
pls2=PLSRegression(n_components=9)

for i in range(Ntrain):
    y=np.eye(2)[Y_train[:, i]]
    pls2.fit(X_train, y)
    y_score = pls2.predict(X_test)
    y_predict[:, i]=np.argmax(y_score, axis=1)
y_true=Y_test
hamming_loss=sklm.hamming_loss(y_true, y_predict)
#mi_auc=sklm.roc_auc_score(y_true, test_score, average='micro')
#ma_auc=sklm.roc_auc_score(y_true, test_score, average='macro')
#sa_auc=sklm.roc_auc_score(y_true, test_score, average='samples')
mi_precision=sklm.precision_score(y_true, y_predict, average='micro')
ma_precision=sklm.precision_score(y_true, y_predict, average='macro')
sa_precision=sklm.precision_score(y_true, y_predict, average='samples')
mi_recall=sklm.recall_score(y_true, y_predict, average='micro')
ma_recall=sklm.recall_score(y_true, y_predict, average='macro')
sa_recall=sklm.recall_score(y_true, y_predict, average='samples')
mi_f1=sklm.f1_score(y_true, y_predict, average='micro')
ma_f1=sklm.f1_score(y_true, y_predict, average='macro')
sa_f1=sklm.f1_score(y_true, y_predict, average='samples')
print(' hamming_loss = %.2f%% '% (100.*hamming_loss))
#print(" mi-auc = %.2f%%, ma-auc =%.2f%%, sa-auc=%.2f%% "
              #% ( 100.* mi_auc, 100.*ma_auc, 100.*sa_auc))
print(" mi-precision = %.2f%%, mi-recall =%.2f%%, mi-f1_score=%.2f%% "
              % ( 100.* mi_precision, 100.*mi_recall, 100.*mi_f1))
print(" ma-precision = %.2f%%, ma-recall =%.2f%%, ma-f1_score=%.2f%% "
              % ( 100.* ma_precision, 100.*ma_recall, 100.*ma_f1))
print(" sa-precision = %.2f%%, sa-recall =%.2f%%, sa-f1_score=%.2f%% "
              % ( 100.* sa_precision, 100.*sa_recall, 100.*sa_f1))
np.savetxt('_TestY_' + str(Nsample) + '_' + str(Nsnr) + 'db', Y_test, delimiter=',')
np.savetxt('_PredY_' + str(Nsample) + '_' + str(Nsnr) + 'db', y_predict, delimiter=',')
print("%s", str(datetime.now()))



