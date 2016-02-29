# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,os,caffe,numpy as np,h5py
from os.path import join
from os import system

def test(net_file,model_topdir,predict_file,gpunum,trialnum,outdir,keyword,outputlayer):

    caffe.set_device(gpunum)
    caffe.set_mode_gpu()

    best_trial,best_iter = getBestRunAll(model_topdir,trialnum,'train.err',keyword)
    model_dir = join(model_topdir,'trial'+str(best_trial))
    model_file = os.path.join(model_dir,'train_iter_'+best_iter+'.caffemodel')
    system(' '.join(['cp',model_file,join(outdir,'bestiter.caffemodel')]))
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
            prob = np.vstack(np.asarray(out[outputlayer]))
            for out in prob:
                f.write('%s\n' % '\t'.join([str(x) for x in out]))

def getBestRunAll(modeltopdir,trialnum,logfile,keyword):
    best_trial = -1
    best_iter = -1
    best_metric = ''
    for trial in range(trialnum):
        modeldir = join(modeltopdir,'trial'+str(trial))
        t_iter,t_metric = getBestRun(modeldir,logfile,keyword)
        if best_metric == '' or t_metric > best_metric:
            best_metric = t_metric
            best_trial = trial
            best_iter = t_iter
    return (best_trial,best_iter)

def getBestRun(modeldir,logfile,keyword):
    with open(os.path.join(modeldir,logfile),'r') as f:
        data = [x for x in f]

    pick = [i for i in range(len(data)) if 'Testing net' in data[i]]
    iter_cnt = []
    metric_cnt = []
    for i in pick[1:]:
        x = data[i].split(' ')
        idx = x.index('Iteration')+1
        iter_cnt.append(x[idx].split(',')[0])
        flag = False
        j = i
        while not flag:
            j = j +1
            if 'Iteration' in data[j]:
                print 'Can\'t find the target metric:',keyword
                sys.exit(1)
            if keyword in data[j]:
                x = data[j].split(' ')
                idx = x.index(keyword)+2
                metric_cnt.append(float(x[idx]))
                flag  = True
    if keyword == 'accuracy':
        return (iter_cnt[np.argmax(metric_cnt)],max(metric_cnt))
    else:
        return (iter_cnt[np.argmin(metric_cnt)],-min(metric_cnt))
