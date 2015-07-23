import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import caffe,os

p_dir = '/media/bigdrive1/zeng/model/RELA_GM12878_ENCSR000EAG_encode_peak_train.2direc.klen5.s200w20c0n20sa0.1.RELA_GM12878_ENCSR000EAG_encode_peak.flank185.klen5'
proto = os.path.join(p_dir, 'deploy.prototxt')
model = os.path.join(p_dir, 'train_iter_20000.caffemodel')
plotout = 'test'
plotformat = 'png'
net = caffe.Net(proto,model,caffe.TEST)

def vis_square(data, plotout,padsize,padval,plotformat):
    data -= data.min()
    data /= data.max()
    # tile the filters into an image
    shape = data.shape
    data = data.reshape(shape[0],shape[1],shape[3]).transpose(0,2,1)


    padding = ((0, 0), (0, padsize), (0,0))
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]) + data.shape[4:])
    print data.shape
    data[200:210,:] = 1
    plt.imshow(data[200:300,:])
    plt.savefig(plotout+'.'+plotformat, dpi=300)
    #for dim in range(data.shape[2]):
    #    plt.plot(data[:,:,dim])
    #    plt.savefig(plotout+'.dim'+str(dim)+'.'+plotformat, dpi=100)


filters = net.params['conv1'][0].data
vis_square(filters,plotout,1,0,plotformat)
