from os.path import join
import numpy as np

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
    with open(join(modeldir,logfile),'r') as f:
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
