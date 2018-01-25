import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from datetime import datetime

RANDOM_SEED = 193
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, name):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=name)


def forwardprop(X, w_1, b_1, w_2, b_2, keep_prob):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    drop_out=tf.nn.dropout(h, keep_prob)
    yhat=tf.add(tf.matmul(drop_out,w_2), b_2)
    return yhat


def forwardprop_score(X, w_1, b_1, w_2, b_2):
    h=tf.nn.relu(tf.add(tf.matmul(X,w_1), b_1))
    yhat=tf.add(tf.matmul(h,w_2), b_2)
    return yhat


def Tlinear(S, w_t):
    w=tf.transpose([w_t[1:,0]])
    tl=tf.add(tf.matmul(S, w), w_t[0,0])
    return tl


def Threshold(y_true, score):
    nt,lt=score.shape
    thresh=np.zeros(nt)
    f1t=0
    for i in range(nt): #range(nt):
        #t_candidate=np.array([0,1])
        #t_candidate=np.append(t_candidate, score[i,:])
        t_candidate=score[i, :]
        #for j in range(lt+2):
        for j in range(lt):
            f1t_j=sklm.f1_score(y_true[i, :], Predict(score[i,:], t_candidate[j]), average='micro')
            if f1t_j>f1t:
                f1t=f1t_j
                thresh[i]=t_candidate[j]
    return thresh


def Predict(xscore, threshold):
    lp=len(xscore)
    pred=np.ones(lp)
    for j in range(lp):
        if xscore[j]<threshold:
            pred[j]=0
    return pred

Nsample=100000
#snr=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
snr=[50]
train_num=2000

print("%s", str(datetime.now()))
for Nsnr in snr:
    print('Nsnr = '+str(Nsnr))
    path = '/home/machinelearningstation/PycharmProjects/multiclassification'
    df0 = pd.read_csv(path + '/data/spec_C2H6_all_9_pxl_1000_samples_' + str(Nsample) + '_SNR_'
                      + str(Nsnr) + '_dB.csv')

    df=df0.sample(frac=1)
    target = (df[df.columns[-9:]].apply(lambda x: x!=0)). astype(int)
    data=df.drop(df.columns[-9:],axis=1)
    print(data.shape)

    all_X=data

    all_Y=pd.DataFrame.as_matrix(target)
    train_size=0.8
    train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=1-train_size, random_state=RANDOM_SEED)
    train_X = train_X[0:train_num]
    train_Y = train_Y[0:train_num]

    x_size=train_X.shape[1]
    h_size=3000
    y_size=train_Y.shape[1]
    t_size=1

    X=tf.placeholder("float", shape=[None, x_size], name='X')
    Y=tf.placeholder("float", shape=[None, y_size], name='Y')
    S=tf.placeholder('float', shape=[None, y_size], name='S')
    T=tf.placeholder('float', shape=[None, t_size], name='T')
    keep_prob=tf.placeholder(tf.float32)

    w_1=init_weights((x_size,h_size), 'w_1')
    b_1=init_weights((1,h_size),'b_1')
    w_2=init_weights((h_size,y_size), 'w_2')
    b_2=init_weights((1,y_size),'b_2')
    w_t=init_weights((y_size+1, t_size), 'w_t')

    yhat=forwardprop(X, w_1, b_1, w_2, b_2, 1)
    score=forwardprop_score(X, w_1,b_1, w_2,b_2)
    t=Tlinear(S, w_t)

    cost_s=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=yhat))
    cost_t=tf.reduce_mean(tf.pow(Tlinear(S, w_t)-T, 2))

    updates_s=tf.train.AdamOptimizer(0.01).minimize(cost_s, name='Adam_LabelScores')
    updates_t=tf.train.AdamOptimizer(0.01).minimize(cost_t, name='Adam_Thresholds')

    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)

    J0s = 0.00
    J1s = 10.00
    epoch=1
    tolerance=1e-8
    while tolerance <= abs(J0s - J1s) and epoch<10000:
        J0s=J1s
        for i in range(len(train_X)/10000):
            sess.run(updates_s,feed_dict={X:train_X[10000*i:(i+1)*10000],Y:train_Y[10000*i:(i+1)*10000]})

        J1s=sess.run(cost_s, feed_dict={X:train_X[0:10000],Y:train_Y[0:10000]})
        print('score epoch= '+str(epoch)+', loss='+str(J1s))
        epoch+=1

    indx = 0
    batch_size=10000
    train_score = np.zeros((len(train_X), y_size))
    while indx < len(train_X):
        train_score[indx:indx + batch_size] = sess.run(score, feed_dict={X: train_X[indx:indx + batch_size],
                                                                         Y: train_Y[indx:indx + batch_size]})
        indx += batch_size

    train_t=Threshold(train_Y, train_score).reshape((train_X.shape[0], t_size))

    J0t=0.00
    J1t=10.00
    epoch=1
    tolerance=1e-14
    while abs(J1t-J0t)>=tolerance and epoch<5000:
        J0t=J1t
        for i in range(len(train_X)/10000):
            sess.run(updates_t,feed_dict={S:train_score[10000*i:(i+1)*10000],T:train_t[10000*i:(i+1)*10000]})

        J1t=sess.run(cost_t, feed_dict={S:train_score[0:10000],T:train_t[0:10000]})
        epoch += 1
        print('thresh epoch= '+str(epoch)+', loss='+str(J1t))
    test_score=sess.run(score, feed_dict={X: test_X,Y: test_Y})
    test_t=sess.run(t, feed_dict={S:test_score})
    y_predict=np.zeros((len(test_score),y_size))
    for i in range(len(test_score)):
        y_predict[i]=Predict(test_score[i], test_t[i])
    y_true = test_Y

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
    np.savetxt('_TestY_' + str(Nsample) + '_' + str(Nsnr) + 'db', test_Y, delimiter=',')
    np.savetxt('_PredY_' + str(Nsample) + '_' + str(Nsnr) + 'db', y_predict, delimiter=',')

    sess.close()

    print("%s", str(datetime.now()))

""""
print("mean(precision)= %.2f%%, mean(recall)= %.2f%%, mean(f1_score)= %.2f%%"
      % ( 100.*np.mean(all_precision), 100.*np.mean(all_recall), 100.*np.mean(all_f1)))
print("std(precision)= %.4f, std(recall)= %.4f, std(f1_score)= %.4f"
      % ( np.std(all_precision), np.std(all_recall), np.std(all_f1)))
"""