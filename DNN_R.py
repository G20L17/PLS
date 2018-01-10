import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import sklearn.metrics as sklm
import matplotlib.pyplot as plt
from datetime import datetime

RANDOM_SEED = 193
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)


def forwardprop(X, w_1, b_1, w_2, b_2, rate, train):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    drop_out=tf.layers.dropout(h, rate=rate, training=train)
    yhat=tf.add(tf.matmul(drop_out,w_2), b_2)
    return yhat

Nsample=100000
snr=[100]#, 1, 2,3,5,10,20,30,50,100,200,300,500]
all_mse=[]
for Nsnr in snr:
    print ('snr = '+str(Nsnr))
    path='/home/machinelearningstation/PycharmProjects/multiclassification'
    df=pd.read_csv(path+'/data/spec_C2H6_all_9_pxl_1000_samples_'+str(Nsample)+'_SNR_'
               +str(Nsnr)+'_dB.csv')
    mse=[]
    print("%s", str(datetime.now()))

    target = df[df.columns[-9:]]

    data = df.drop(df.columns[-9:], axis=1)

    all_X=pd.DataFrame.as_matrix(data)

    all_Y=pd.DataFrame.as_matrix(target)
    train_size=0.8
    train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=1-train_size, random_state=RANDOM_SEED)
    #kf=KFold(n_splits=5)
    #fold_ind=1
    beta=0.01
    for i in range(1):
    #for train, test in kf.split(all_Y):
        #print('Fold '+str(fold_ind))
        #fold_ind+=1
        #train_X, test_X, train_Y, test_Y = all_X[train], all_X[test], all_Y[train], all_Y[test]
        x_size=train_X.shape[1]
        h_size=3000
        y_size=train_Y.shape[1]


        X=tf.placeholder("float", shape=[None, x_size], name='X')
        Y=tf.placeholder("float", shape=[None, y_size], name='Y')

        w_1=init_weights((x_size,h_size), 'w_1')
        b_1=init_weights((1,h_size),'b_1')
        w_2=init_weights((h_size,y_size), 'w_2')
        b_2=init_weights((1,y_size),'b_2')

        yhat=forwardprop(X, w_1, b_1, w_2, b_2, 1,False)
        y_pred=forwardprop(X, w_1, b_1, w_2, b_2, 1, False)

        cost = tf.reduce_mean(tf.square(yhat - Y), name='loss_function')
        regularizer=tf.nn.l2_loss(w_1)+tf.nn.l2_loss(w_2)
        cost = tf.reduce_mean(cost+beta*regularizer)
        updates = tf.train.AdamOptimizer(0.01).minimize(cost, name='AdamOptimizer')
        cost_test=tf.reduce_mean(tf.square(y_pred-Y))
        sess=tf.Session()
        init=tf.global_variables_initializer()
        sess.run(init)

        J0s = 0.00
        J1s = 10.00
        epoch=1
        tolerance=1e-24
        batch_size=10000

        while tolerance <= abs(J0s - J1s) and epoch<131:
            J0s=J1s
            ind_start = 0
            while ind_start<len(train_X):

                sess.run(updates,feed_dict={X:train_X[ind_start:ind_start+batch_size]
                    ,Y:train_Y[ind_start:ind_start+batch_size]})
                ind_start+=batch_size

            J1s=sess.run(cost, feed_dict={X:train_X[0:batch_size],Y:train_Y[0:batch_size]})
            epoch += 1
            print('score epoch= '+str(epoch)+', loss='+str(J1s))

        pred_Y=sess.run(y_pred, feed_dict={X: test_X})

        #fold_mse=sklm.mean_squared_error(test_Y, pred_Y)
        fold_mse=sess.run(cost_test, feed_dict={X:test_X, Y: test_Y})
        print(fold_mse)
        mse.append(fold_mse)
        all_mse.append(fold_mse)
        sess.close()
        #print("%s", str(datetime.now()))
        np.savetxt('Fold '+str(i+1)+'_TestY_' + str(Nsample) + '_' + str(Nsnr) + 'db', test_Y, delimiter=',')
        np.savetxt('Fold '+str(i+1)+'_PredY_' + str(Nsample) + '_' + str(Nsnr) + 'db', pred_Y, delimiter=',')

    print("%s", str(datetime.now()))


    #print(mse)

    np.savetxt('MSE_' + str(Nsample) + '_' + str(Nsnr) + 'db', mse, delimiter=',')
    #print('mean mse = '+str(np.mean(mse))+', std mse = '+str(np.std(mse)))
    print('--------------------------------------')
np.savetxt('MSE_' + str(Nsample) + '_all_db', all_mse, delimiter=',')
""""
print("mean(precision)= %.2f%%, mean(recall)= %.2f%%, mean(f1_score)= %.2f%%"
      % ( 100.*np.mean(all_precision), 100.*np.mean(all_recall), 100.*np.mean(all_f1)))
print("std(precision)= %.4f, std(recall)= %.4f, std(f1_score)= %.4f"
      % ( np.std(all_precision), np.std(all_recall), np.std(all_f1)))
"""
