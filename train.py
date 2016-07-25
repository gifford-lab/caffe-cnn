# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,caffe,os
from os.path import join

solver_file = sys.argv[1]
outdir = sys.argv[2]
gpunum = int(sys.argv[3])

sys.stdout = open(join(outdir,'train.out'), 'w')
sys.stderr = open(join(outdir,'train.err'), 'w')

caffe.set_device(gpunum)
caffe.set_mode_gpu()

solver = caffe.get_solver(solver_file)
solver.solve()

