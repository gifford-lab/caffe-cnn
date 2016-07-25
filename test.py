# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py,cPickle
from os.path import join,dirname,exists,realpath
from os import system,makedirs

def test(net_file,model_file,predict_file,gpunum,outdir,outputlayer):
    caffe.set_device(gpunum)
    caffe.set_mode_gpu()
    if not exists(outdir):
        makedirs(outdir)
    outfile = os.path.join(outdir,'bestiter.pred')
    outputlayer_split = outputlayer.split('_')
    outputlayer_cnt = len(outputlayer_split)
    flag = False
    outdata = []

    net = caffe.Net(realpath(net_file), realpath(model_file),caffe.TEST)
    with open(predict_file,'r') as f:
        files = [x.strip() for x in f]

    with open(outfile,'w') as f:
        for batchfile in files:
            fi    = h5py.File(batchfile, 'r')
            dataset = np.asarray(fi['data'])
            out = net.forward_all(data=dataset,blobs=outputlayer_split)
            for i in range(outputlayer_cnt):
                if not flag:
                    outdata.append( np.vstack(np.asarray(out[outputlayer_split[i]])) )
                else:
                    outdata[i] = np.vstack((outdata[i],np.vstack(np.asarray(out[outputlayer_split[i]]))))
            flag = True
        for out in outdata[0]:
            f.write('%s\n' % '\t'.join([str(x) for x in out]))

    with open(join(outdir,'bestiter.pred.params.pkl'),'wb') as f:
        cPickle.dump((outdata,outputlayer_split),f,protocol=cPickle.HIGHEST_PROTOCOL)






































