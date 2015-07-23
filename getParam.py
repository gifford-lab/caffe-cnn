import numpy as np
import caffe,os,h5py

p_dir = '/media/bigdrive1/zeng/model/RELA_GM12878_ENCSR000EAG_encode_peak_train.2direc.klen5.s200w20c0n20sa0.1.RELA_GM12878_ENCSR000EAG_encode_peak.flank185.klen5'
proto = os.path.join(p_dir, 'deploy.prototxt')
model = os.path.join(p_dir, 'train_iter_20000.caffemodel')
outfile = os.path.join(p_dir,'train_iter_20000.param')
net = caffe.Net(proto,model,caffe.TEST)

filters = net.params['conv1'][0].data

comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(outfile, 'w') as f:
    f.create_dataset('filters', data=filters, **comp_kwargs)

