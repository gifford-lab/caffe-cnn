#!/usr/bin/env python
import sys
import os

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

    assert 'order' in params.keys()
    assert 'model_name' in params.keys()
    assert 'gpu_num' in params.keys()
    assert 'iter_num_to_test' in params.keys()
    assert 'data_src' in params.keys()

    print '####### Parameters Input #######'
    print 'order: ' + params['order']
    print 'model_name: ' + params['model_name']
    print 'gpu_num: ' + params['gpu_num']
    print 'iter_num_to_test: ' + params['iter_num_to_test']
    print 'data_src: ' + params['data_src']
    print '################################'

    currentdir = os.path.abspath('.')
    dirname = os.path.abspath('/media/bigdrive1/zeng/model/' + params['model_name'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cmd = 'cp solver.prototxt ' + dirname + '/'
    os.system(cmd)
    cmd = 'cp train_val.prototxt ' + dirname + '/'
    os.system(cmd)
    cmd = 'cp deploy.prototxt ' + dirname + '/'
    os.system(cmd)

    os.chdir(dirname)

    flag = False;
    if params['order']=='getdata':
        cmd = 'scp -r ' + params['data_src']  + ' .'
        flag = True

    if params['order']=='train':
        cmd = 'python ' + currentdir + '/train.py solver.prototxt ' +  params['gpu_num'] + ' 1> train.out 2> train.err'
        flag = True

    if params['order']=='valid':
        cmd = 'python ' + currentdir + '/valid.py  train_val.prototxt  train_iter_' + params['iter_num_to_test'] + '.caffemodel ' + params['test_iter'] + ' ' + params['gpu_num'] + ' 1> valid_iter' + params['iter_num_to_test'] + '.out 2> valid_iter' + params['iter_num_to_test'] + '.err'
        flag = True
    if params['order']=='test':
        cmd = 'python ' + currentdir + '/test.py deploy.prototxt train_iter_' + params['iter_num_to_test'] + '.caffemodel ' + params['topredict'] + ' ' + params['predictoutput'] + ' ' + params['gpu_num']
        flag = True
    if params['order']=='test_eval':
        cmd = ' '.join(['python' ,currentdir + '/test_eval.py' , params['predictoutput'],params['topredict'] ])
        flag = True


    if not flag:
        print 'Cannot recognize order; Exit'
        exit(2)

    print 'Running cmd:'
    print cmd
    os.system(cmd)

if __name__ == '__main__':
    main()
