import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp


def extract_file(filename, shape):
    dt = np.dtype(np.float64).newbyteorder('>')
    data = np.ndarray(shape=(0, 0), dtype=dt)
    with open(filename) as f:
        for line in f:
            data = np.append(data, np.array(line.split(',')).astype(dt))
    data=data.reshape(shape)
    return data

Nsample=100000
Nsnr=50
test_size=0.2
path = '/home/machinelearningstation/PycharmProjects/spectrum2/DNN_C/'
pred_file=path+'_PredY_'+str(Nsample)+'_'+str(Nsnr)+'db'
test_file=path+'_TestY_'+str(Nsample)+'_'+str(Nsnr)+'db'

X=extract_file(test_file, (20000, 9))
Y=extract_file(pred_file, (20000, 9))


plt.plot(X[:, 0], Y[:, 0])
plt.show()