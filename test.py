# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys,os
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
import h5py
from sklearn.metrics import roc_curve, auc,accuracy_score

def main():
    if len(sys.argv) != 6:
        print 'Usage: python test.py net_file model_file predict_file outfile gpunum'
        sys.exit(2)

    net_file = sys.argv[1]
    model_file = sys.argv[2]
    predict_file = sys.argv[3]
    outfile =  sys.argv[4]
    gpunum = int(sys.argv[5])

    caffe.set_device(gpunum)
    caffe.set_mode_gpu()

    net = caffe.Net(net_file, model_file,caffe.TEST)
    predict_dir = os.path.dirname(os.path.dirname(predict_file))
    with open(predict_file,'r') as f:
        files = [os.path.join(predict_dir,x.strip()) for x in f]

    with open(outfile,'w') as f:
        for batchfile in files:
            fi    = h5py.File(batchfile, 'r')
            dataset = np.asarray(fi['data'])
            out = net.forward_all(data=dataset)
            prob = np.vstack(np.asarray(out['prob']))
            for out in prob:
                f.write('%s\n' % '\t'.join([str(x) for x in out]))

if __name__ == '__main__':
    main()
