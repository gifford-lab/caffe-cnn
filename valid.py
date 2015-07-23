# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from sklearn.metrics import roc_curve, auc,accuracy_score

def main():
    if len(sys.argv) != 5:
        print 'Usage: python test.py net_file model_file test_iters_num'
        sys.exit(2)

    net_file = sys.argv[1]
    model_file = sys.argv[2]
    test_iters = int(sys.argv[3])
    gpunum = int(sys.argv[4])

    caffe.set_device(gpunum)
    caffe.set_mode_gpu()

    net = caffe.Net(net_file, model_file,caffe.TEST)

    all_pred = [None] * test_iters
    all_label = [None] * test_iters
    accuracy = 0
    for i in range(test_iters):
        net.forward()
        pred=net.blobs['fc1'].data[...]
        all_pred[i] = pred[:,1]
        all_label[i]=net.blobs['label'].data[...]
        accuracy += net.blobs['accuracy'].data

    accuracy /= test_iters
    print 'accuracy: %f' % accuracy

    #all_pred = np.asarray(np.concatenate(all_pred))
    #all_label = np.asarray(np.concatenate(all_label))
    #fpr, tpr, thresholds = roc_curve(all_label,all_pred)
    #roc_auc = auc(fpr, tpr)
    #print 'auc: %f' % roc_auc

    #thresh = 0
    #correct = [i for i in range(len(all_pred)) if all_pred[i]>thresh and all_label[i]==1 or all_pred[i]<thresh and all_label[i]==0]
    #man_accuracy = float(len(correct))/len(all_pred)
    #print 'manually calculated accuracy: %f' % man_accuracy
if __name__ == '__main__':
    main()
