# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe


def main():
    if len(sys.argv) != 2:
        print 'Usage: python train.py solver_file GPUnum'
        sys.exit(2)
    solver_file = sys.argv[1]
    caffe.set_mode_gpu()

    solver = caffe.get_solver(solver_file)
    solver.solve()

if __name__ == '__main__':
    main()
