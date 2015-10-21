# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py,cPickle
from os.path import join,exists
from os import makedirs,system

topdir = sys.argv[1]
predict_file = sys.argv[2]
gpunum = int(sys.argv[3])
batchsize = int(sys.argv[4])
kernelnum = int(sys.argv[5])
kernelsize = int(sys.argv[6])

def getBestRun(t_dir):
    protofile = join(t_dir,'mri-best','best_trial','deploy.prototxt')
    with open(join(t_dir,'mri-best','best_trial','bestiter.info'),'r') as f:
        f.readline()
        bestrun = f.readline().strip().split()
    modelfile = join(t_dir,'mri-best','trial'+bestrun[0],'train_iter_'+bestrun[1]+'.caffemodel')
    outdir = join(t_dir,'filters')
    if exists(outdir):
        system('rm -r '+outdir)
    makedirs(outdir)
    outfile = join(outdir,'filter.pkl')
    return (protofile,modelfile,outfile)

caffe.set_device(gpunum)
caffe.set_mode_gpu()

net_file,model_file,outfile = getBestRun(topdir)

net = caffe.Net(net_file, model_file,caffe.TEST)

predict_dir = os.path.dirname(predict_file)
with open(predict_file,'r') as f:
    files = [x.strip() for x in f]

activator = [[] for x in range(kernelnum)]
with open(outfile,'w') as f:
    for batchfile in files:
        fi    = h5py.File(batchfile, 'r')
        dataset = np.asarray(fi['data'])
        accum = 0
        d_size = len(dataset)
        while accum+batchsize < d_size:
            e = np.min([(accum+batchsize),d_size])
            d = dataset[accum:e]
            out = net.forward(data=d,end='conv1')['conv1']
            for i in range(len(out)):
                for k in range(kernelnum):
                    t_out = out[i][k][0]
                    t_data = d[i]
                    if np.max(t_out)>0:
                        maxpos = np.argmax(t_out)
                        data_s = np.max([0,maxpos-np.floor(kernelsize/2)])
                        data_e = np.min([len(t_data[0][0])-1,maxpos+np.floor((kernelsize-1)/2)])
                        t_ans = np.squeeze(np.transpose(t_data[:,:,data_s:(data_e+1)],(2,0,1)))
                        if len(t_ans) < kernelsize:
                            padding = np.asarray([[0,0,0,0] for x in range(kernelsize - len(t_ans))])
                            if data_s == 0:
                                t_ans = np.concatenate((padding,t_ans))
                            else:
                                t_ans = np.concatenate((t_ans,padding))
                        activator[k].append(t_ans)
            accum = e

with open(outfile,'w') as f:
    cPickle.dump(activator,f)
