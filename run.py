#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import sys,os
from time import localtime, strftime
from helper import getBestRunAll
from test import test
from test_eval import test_eval
from os import system,makedirs,chdir,getcwd
from os.path import join,abspath,exists,realpath,dirname,basename

def main():
    if len(sys.argv)!=3:
        print 'Usage: python run.py order'
        sys.exit(2)

    fcwd = dirname(realpath(__file__))
    order = sys.argv[1]
    paramfile = sys.argv[2]

    with open(paramfile,'r') as f:
        paramdata = f.readlines()

    params = {}
    params['order'] = order
    params['predict_filelist'] = 'data/test.txt'
    for i in range(len(paramdata)):
        line = paramdata[i].strip().split()
        params.update({line[0]:line[1]})

    print '####### Parameters Input #######'
    for key in params.keys():
        print key + ': ' + params[key]
    print '################################'

    for code in ['output_topdir','gpunum','caffemodel_topdir','model_name','batch_name','batch2predict','optimwrt','outputlayer']:
        assert(code in params.keys())

    ctime = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    datadir = abspath(join(params['output_topdir'],'data'))

    params['solver_file'] = join(params['caffemodel_topdir'],'solver.prototxt')
    params['trainval_file'] = join(params['caffemodel_topdir'],'trainval.prototxt')
    params['deploy_file'] = join(params['caffemodel_topdir'],'deploy.prototxt')

    flag = False;

    if params['order']=='train':
        if not 'batch_name' in params.keys():
            params['batch_name'] = ctime

        modeltopdir = abspath(join(params['output_topdir'],params['model_name'],params['batch_name']))
        if exists(modeltopdir):
            print 'model folder exists, will remove'
            system('rm -r ' + modeltopdir)
        makedirs(modeltopdir)

        cwd = getcwd()
        for trial in range(int(params['trial_num'])):
            modeldir = join(modeltopdir,'trial'+str(trial))
            makedirs(modeldir)
            system(' '.join(['cp ',params['solver_file'], modeldir]))
            system(' '.join(['cp ',params['trainval_file'], modeldir]))

            chdir(modeldir)
            outlog = join(modeldir,'train.out')
            errlog = join(modeldir,'train.err')
            system(' '.join(['python',join(fcwd,'train.py'),basename(params['solver_file']),modeldir,params['gpunum'],'1 >',outlog,'2>',errlog]))
            chdir(cwd)

        ### Get the best performing iteration among all the trials
        best_trial,best_iter = getBestRunAll(modeltopdir,int(params['trial_num']),'train.err',params['optimwrt'])
        model_file = join(modeltopdir,'trial'+str(best_trial),'train_iter_'+best_iter+'.caffemodel')
        testdir = join(modeltopdir,'best_trial')
        if exists(testdir):
            print 'testdir '+testdir+' exists, will be removed'
            system('rm -r ' + testdir)
        makedirs(testdir)
        system(' '.join(['cp',model_file,join(testdir,'bestiter.caffemodel')]))
        system(' '.join(['cp',params['deploy_file'], testdir]))
        with open(join(testdir,'bestiter.info'),'w') as f:
	    f.write('best_trial\tbest_iter\n')
	    f.write('%d\t%s\n' % (best_trial,best_iter))
	flag = True

    if params['order']=='test':
        modeltopdir = abspath(join(params['output_topdir'],params['model_name'],params['batch2predict']))
        testdir = join(modeltopdir,'best_trial')
        test(params['deploy_file'],join(testdir,'bestiter.caffemodel'),join(params['output_topdir'],params['predict_filelist']),int(params['gpunum']),testdir,params['outputlayer'])
        flag = True

    if params['order']=='test_eval':
        pred_topdir = abspath(join(params['output_topdir'],params['model_name'],params['batch2predict'],'best_trial'))
        pred_f = join(pred_topdir,'bestiter.pred')
        real_f = join(params['output_topdir'],params['predict_filelist'])
        outfile = join(pred_topdir,'bestiter.pred.eval')
        test_eval(pred_f,real_f,outfile)
        flag = True

    if params['order'] == 'pred':
        for code in ['deploy2predictW','caffemodel2predictW','data2predict','predict_outdir']:
            assert(code in params.keys())
        test(params['deploy2predictW'],params['caffemodel2predictW'],params['data2predict'],int(params['gpunum']),params['predict_outdir'],params['outputlayer'])
        flag = True

    if not flag:
        print 'Cannot recognize order; Exit'
        exit(2)

if __name__ == '__main__':
    main()
