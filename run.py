#!/usr/bin/env python
import sys,os
from time import localtime, strftime

def main():
    if len(sys.argv)!=2:
        print 'Usage: python run.py order'
        sys.exit(2)

    order = sys.argv[1]

    with open('param.list') as f:
        paramdata = f.readlines()

    params = {}
    params['order'] = order
    for i in range(len(paramdata)):
        line = paramdata[i].strip().split()
        params.update({line[0]:line[1]})

    assert 'model_topdir' in params.keys()
    assert 'predict_model' in params.keys()
    assert 'predict_filelist' in params.keys()
    assert 'predict_out' in params.keys()
    assert 'predict_gpu' in params.keys()
    assert 'order' in params.keys()
    assert 'model_name' in params.keys()
    assert 'data_src' in params.keys()

    print '####### Parameters Input #######'
    print 'order: ' + params['order']
    print 'model_name: ' + params['model_name']
    print 'data_src: ' + params['data_src']
    print 'model_topdir: ' + params['model_topdir']
    print 'predict_model: ' + params['predict_model']
    print 'predict_filelist: ' + params['predict_filelist']
    print 'predict_out: ' + params['predict_out']
    print 'predict_gpu: ' + params['predict_gpu']
    print '################################'

    currentdir = os.path.abspath('.')
    ctime = strftime("%Y-%m-%d-%H-%M-%S", localtime())

    datadir = os.path.abspath(os.path.join(params['model_topdir'],params['model_name'],'data'))

    flag = False;
    if params['order']=='getdata':
        if os.path.exists(datadir):
            print 'data folder exists, will remove'
            os.system('rm -r '+datadir)
        os.makedirs(datadir)

        cmd = ' '.join(['scp -r ',params['data_src']+'/*',datadir])
        flag = True

    if params['order']=='train':
        modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['model_name'],ctime))
        if not os.path.exists(modeldir):
            print 'model folder exists, will remove'
            os.makedirs(modeldir)
        cmd = ' '.join(['cp ',os.path.join(currentdir,'solver.prototxt'), modeldir])
        os.system(cmd)
        cmd = ' '.join(['cp ',os.path.join(currentdir,'train_val.prototxt'), modeldir])
        os.system(cmd)

        cmd = ' '.join(['python ',os.path.join(currentdir,'train.py'), 'solver.prototxt', '1> train.out 2> train.err'])
        flag = True

    if params['order']=='test':
        modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['model_name'],params['predict_model']))
        cmd = ' '.join(['cp ',os.path.join(currentdir,'deploy.prototxt'), modeldir])
        os.system(cmd)

        os.chdir(modeldir)
        cmd = ' '.join(['python', os.path.join(currentdir,'test.py'), 'deploy.prototxt',params['predict_iter'],params['predict_filelist'],params['predict_out'],params['predict_gpu']])
        flag = True

    if params['order']=='test_eval':
        modeldir = os.path.abspath(os.path.join(params['model_topdir'],params['model_name'],params['predict_model']))
        os.chdir(modeldir)
        cmd = ' '.join(['python' ,os.path.join(currentdir,'test_eval.py') , params['predict_out'],params['predict_filelist'] ])
        flag = True

    if not flag:
        print 'Cannot recognize order; Exit'
        exit(2)

    print 'Running cmd:'
    print cmd
    os.system(cmd)

if __name__ == '__main__':
    main()
