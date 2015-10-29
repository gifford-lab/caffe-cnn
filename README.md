A platform for training and testing convolutional neural network (CNN) based on [Caffe](http://caffe.berkeleyvision.org/)

## Prerequisite
### Install Caffe and pycaffe
Following [instruction](http://caffe.berkeleyvision.org/installation.html) to install Caffe and pycaffe.

### Prepare data
All the data should be put under $model_topdir$/data/, where `model_topdir` is the top folder of the output. 

In the near future, we will provide the user with a script to generate all the data needed for CNN application on 1-D genomic sequence.

#### Actual data
Training, validating and testing data should be prepared as 4-D matrix with dimension _iteration * channel * width * height_ in HDF5 format.

In deep learning, the data size are usually enormous, making it hard to load the datasets into memory. A good practice is to partition the dataset into small batches. 

#### Data manifest
For each of train/valid/test set, we require the user to provide a txt file  to specify the ABSOLUTE PATH of all the HDF5 batches.


For example if we have the following TRAINING data files under $model_topdir$/data/

+ train.batch0.hd5
+ train.batch1.hd5
+ ... (for more batches)

Then the content of train.txt should be :

```
/$model_topdir$/data/train.batch0.hd5
/$model_topdir$/data/train.batch1.hd5
...
```

In sum the files under /$model_topdir$/data/ are:

+ train.txt
+ valid.txt
+ test.txt
+ train.batch0.hd5
+ train.batch1.hd5
+ ... (for more batches)
+ valid.batch0.hd5
+ valid.batch1.hd5
+ ... (for more batches)
+ test.batch0.hd5
+ test.batch1.hd5
+ ... (for more batches)


### Prepare model files for Caffe
Refer to [here](http://caffe.berkeleyvision.org/) and [here](https://github.com/BVLC/caffe/tree/master/models) for instructions and examples in generating the following three files: 

(as exemplified under example/)

+ trainval.prototxt
+ solver.prototxt
+ deploy.prototxt

Note that in trainval.prototxt, data should be input through hdf5 layer. And the data source (for training as an example) should be either $../../../data/train.txt$ or absolute path $model_topdir$/data/train.txt.

### Prepare param.list
This file specificies all the parameters in the model. 

Example : (example/param.list)

```
model_topdir /cluster/output 
data_src /cluster/originaldata
gpunum 0
solver_file example/solver.prototxt
trainval_file example/trainval.prototxt
deploy_file example/deploy.prototxt
codedir CAFFECNN_ROOT
modelname basicmodel
model_batchname trail0
predictmodel_batch trail0
predict_filelist data/test.txt
trial_num 3
```

+ `model_topdir`: The top output folder
+ `data_src`: not used in current version
+ `gpunum`: The GPU device number to run on
+ `solver_file`: The path to solver file
+ `trainval_file`: The path to trainval file
+ `deploy_file`: The path to deploy file
+ `codedir`: The directory of this repo
+ `modelname`: The name of the model. 
+ `model_batchname`: The name of the specific model version. All the output will be therefore generated under $model_topdir$/$modelname$/$model_batchname$/, i.e. /cluster/output/basicmodel/trail0/ in the example.
+ `predictmodel_batch`: The model version to test on
+ `predict_filelist`: The path of the test HDF5 file relative to `model_topdir` 
+ `trial_num`: The number of training trial.


## Run the model


#### Train
Train a CNN using the parameters provided in `param.list`. $trial_num$ number of independent training will be performed using the same parameters.


```
python run.py train example/param.list
```

#### Test
Test the trained CNN on test set using the model with the best validation accuracy from all the iterations reported of all the $trial_num$ of models trained.


```
python run.py test example/param.list
```

#### Evaluate the testing result
Evaluate the performance on test set by accuracy and area under receiver operating curve (auROC)


```
python run.py test_eval example/param.list
```
