import matplotlib
matplotlib.use('Agg')
import sys,numpy as np
import scipy,h5py,os
from scipy import stats
from sklearn.metrics import roc_curve, auc,accuracy_score

def test_eval(pred_f,real_f,outfile):

    with open(pred_f,'r') as f:
        pred = [np.asarray([float(y) for y in x.strip().split()]).argsort()[-1] for x in f]
        f.seek(0)
        pred_numeric = [np.asarray([float(y) for y in x.strip().split()])[-1] for x in f]

    real = []
    realdir = os.path.dirname(real_f)
    with open(real_f,'r') as f:
        files = [x.strip() for x in f]


    for batchfile in files:
        fi    = h5py.File(os.path.join(realdir,batchfile), 'r')
        real.append(np.asarray(fi['label']))

    real = np.concatenate(np.asarray(real))

    accuracy = np.sum([1 for i in range(len(real)) if real[i]==pred[i]])
    accuracy = accuracy / float(len(real))
    print 'accuracy: %f ' % accuracy

    fpr, tpr, thresholds = roc_curve(real,pred_numeric)
    roc_auc = auc(fpr, tpr)
    print 'auc: %f' % roc_auc

    with open(outfile,'w') as f:
        f.write('Accuracy:%f\n' % accuracy)
        f.write('ROCAUC:%f\n' % roc_auc)
