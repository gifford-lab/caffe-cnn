# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py
from os.path import join

def test(net_file,model_topdir,predict_file,gpunum,trialnum,outdir):

    caffe.set_device(gpunum)
    caffe.set_mode_gpu()

    best_trial,best_iter = getBestRunAll(model_topdir,trialnum,'train.err')
    model_dir = join(model_topdir,'trial'+str(best_trial))
    model_file = os.path.join(model_dir,'train_iter_'+best_iter+'.caffemodel')
    outfile = os.path.join(outdir,'bestiter.pred')

    with open(join(outdir,'bestiter.info'),'w') as f:
	f.write('best_trial\tbest_iter\n')
	f.write('%d\t%s\n' % (best_trial,best_iter))

    net = caffe.Net(net_file, model_file,caffe.TEST)
    predict_dir = os.path.dirname(predict_file)
    with open(predict_file,'r') as f:
        files = [x.strip() for x in f]

    with open(outfile,'w') as f:
        for batchfile in files:
            fi    = h5py.File(batchfile, 'r')
            dataset = np.asarray(fi['data'])
            out = net.forward_all(data=dataset)
            prob = np.vstack(np.asarray(out['prob']))
            for out in prob:
                f.write('%s\n' % '\t'.join([str(x) for x in out]))

def getBestRunAll(modeltopdir,trialnum,logfile):
    best_trial = -1
    best_iter = -1
    best_acc = -1
    for trial in range(trialnum):
        modeldir = join(modeltopdir,'trial'+str(trial))
        t_iter,t_acc = getBestRun(modeldir,logfile)
        if t_acc > best_acc:
            best_acc = t_acc
            best_trial = trial
            best_iter = t_iter
    return (best_trial,best_iter)

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

    return (iter_cnt[np.argmax(acc_cnt)],max(acc_cnt))

