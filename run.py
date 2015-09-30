#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import sys,os
from time import localtime, strftime
#from train import train
from test import test
from test_eval import test_eval

def main():
    if len(sys.argv)!=3:
        print 'Usage: python run.py order'
        sys.exit(2)

    order = sys.argv[1]
    paramfile = sys.argv[2]

    with open(paramfile,'r') as f:
        paramdata = f.readlines()

    params = {}
    params['order'] = order
    for i in range(len(paramdata)):
        line = paramdata[i].strip().split()
        params.update({line[0]:line[1]})

    print '####### Parameters Input #######'
    for key in params.keys():
        print key + ': ' + params[key]
    print '################################'

    ctime = strftime("%Y-%m-%d-%H-%M-%S", localtime())
    datadir = os.path.abspath(os.path.join(params['model_topdir'],'data'))

    flag = False;
    if params['order']=='getdata':
        if os.path.exists(datadir):
            print 'data folder exists, will remove'
            os.system('rm -r '+datadir)
        os.makedirs(datadir)

        cmd = ' '.join(['scp -r ',params['data_src']+'/*',datadir])
        os.system(cmd)
        flag = True

    if params['order']=='train':
        if 'modelname' in params.keys():
            modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['modelname'],params['model_batchname']))
        else:
            modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['modelname'],ctime))

        if os.path.exists(modeldir):
            print 'model folder exists, will remove'
            os.system('rm -r ' + modeldir)
        os.makedirs(modeldir)

        cmd = ' '.join(['ln -s',os.path.join(params['model_topdir'],'data'),os.path.join(params['model_topdir'],params['modelname'],'data')])
        os.system(cmd)

        cmd = ' '.join(['cp ',params['solver_file'], modeldir])
        os.system(cmd)
        cmd = ' '.join(['cp ',params['trainval_file'], modeldir])
        os.system(cmd)

        os.chdir(modeldir)
        #train(os.path.basename(params['solver_file']),modeldir)
        outlog = os.path.join(modeldir,'train.out')
        errlog = os.path.join(modeldir,'train.err')
        os.system(' '.join(['python',os.path.join(params['codedir'],'train.py'),os.path.basename(params['solver_file']),modeldir,params['gpunum'],'1 >',outlog,'2>',errlog]))
        flag = True

    if params['order']=='test':
        modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['modelname'],params['predictmodel_batch']))
        cmd = ' '.join(['cp',params['deploy_file'], modeldir])
        os.system(cmd)

        os.chdir(modeldir)
        test(params['deploy_file'],modeldir,os.path.join(params['model_topdir'],params['predict_filelist']),int(params['gpunum']))
        flag = True

    if params['order']=='test_eval':
        modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['modelname'],params['predictmodel_batch']))
        model = [x for x in os.listdir(modeldir) if os.path.isfile(os.path.join(modeldir,x)) and 'bestiter' in x][0]
        outfile = os.path.join(modeldir,model + '.eval')
        test_eval(os.path.join(modeldir,model),os.path.join(params['model_topdir'],params['predict_filelist']),outfile)
        flag = True

    if not flag:
        print 'Cannot recognize order; Exit'
        exit(2)

if __name__ == '__main__':
    main()
