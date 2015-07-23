import sys,numpy as np
import scipy,h5py,os
from scipy import stats
from sklearn.metrics import roc_curve, auc,accuracy_score

pred_f = sys.argv[1]
real_f = sys.argv[2]

with open(pred_f,'r') as f:
    pred = [np.asarray([float(y) for y in x.strip().split()]).argsort()[-1] for x in f]

real = []
with open(real_f,'r') as f:
    files = [x.strip() for x in f]


for batchfile in files:
    fi    = h5py.File(os.path.join(batchfile), 'r')
    real.append(np.asarray(fi['label']))

real = np.concatenate(np.asarray(real))

accuracy = np.sum([1 for i in range(len(real)) if real[i]==pred[i]])
accuracy = accuracy / float(len(real))
print accuracy
#fpr, tpr, thresholds = roc_curve(real,pred)
#roc_auc = auc(fpr, tpr)
#print 'auc: %f' % roc_auc
