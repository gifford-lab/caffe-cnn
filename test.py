# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py

def test(net_file,model_dir,predict_file,gpunum,outfile=''):

    caffe.set_device(gpunum)
    caffe.set_mode_gpu()

    best_iter = getBestRun(model_dir,'train.err')
    model_file = os.path.join(model_dir,'train_iter_'+best_iter+'.caffemodel')
    if outfile =='':
        outfile = os.path.join(model_dir,'bestiter_'+best_iter+'.pred')

    net = caffe.Net(net_file, model_file,caffe.TEST)
    predict_dir = os.path.dirname(predict_file)
    with open(predict_file,'r') as f:
        files = [os.path.join(model_dir,x.strip()) for x in f]

    with open(outfile,'w') as f:
        for batchfile in files:
            fi    = h5py.File(batchfile, 'r')
            dataset = np.asarray(fi['data'])
            out = net.forward_all(data=dataset)
            prob = np.vstack(np.asarray(out['prob']))
            for out in prob:
                f.write('%s\n' % '\t'.join([str(x) for x in out]))

def getBestRun(modeldir,logfile):
    with open(os.path.join(modeldir,logfile),'r') as f:
        data = [x for x in f]

    pick = [i for i in range(len(data)) if 'Testing net' in data[i]]
    iter_cnt = []
    acc_cnt = []
    for i in pick[1:]:
        x = data[i].split(' ')
        idx = x.index('Iteration')+1
        iter_cnt.append(x[idx].split(',')[0])

        x = data[i+1].split(' ')
        idx = x.index('accuracy')+2
        acc_cnt.append(float(x[idx]))

    return iter_cnt[np.argmax(acc_cnt)]

