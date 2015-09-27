# Make sure that caffe is on the python path:
import matplotlib
matplotlib.use('Agg')
import sys,caffe,os
#def train(solver_file,outdir):
solver_file = sys.argv[1]
outdir = sys.argv[2]

sys.stdout = open(os.path.join(outdir,'train.out'), 'w')
sys.stderr = open(os.path.join(outdir,'train.err'), 'w')

caffe.set_mode_gpu()

solver = caffe.get_solver(solver_file)
solver.solve()

