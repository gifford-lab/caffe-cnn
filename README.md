A platform for training and testing convolutional neural network (CNN) based on [Caffe](http://caffe.berkeleyvision.org/)

## Prerequisite
[Caffe and pycaffe](http://caffe.berkeleyvision.org/installation.html) (already included for mri-wrapper)

## Quick run

First replace all $REPO_HOME$ in the following files as the actual directory of the repository:

+ example/param.list
+ example/data/train.txt
+ example/data/valid.txt
+ example/data/test.txt


#### Train

```
python run.py train example/param.list
```

#### Test

```
python run.py test example/param.list
```

#### Evaluate the testing result

```
python run.py test_eval example/param.list
```

The output will be under /$REPO_HOME$/example/basicmodel/version0/best_trial/


## Data preparation
All the data should be put under $model_topdir$/data/, where `model_topdir` is the top folder of the output. 

An example list of data is [here](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data)

**_For mri-wrapper user, we now provide a python [script](https://github.com/gifford-lab/caffe-cnn/tree/master/embedH5.py) to automatically generate the corresponding data matrix and data manifest file from  1-D genomic sequence in [FASTA](https://github.com/gifford-lab/caffe-cnn/tree/master/example/sequence/sample.fasta) or [TSV](https://github.com/gifford-lab/caffe-cnn/tree/master/example/sequence/sample.tsv) format. The user still has to manually split the sequence data into train, valid and test set before using this script on each one of them. Check out the usage of the script by:_**
 
 ```
 python embedH5.py -h
 ```

#### Data matirx
Training, validating and testing data should be each prepared as 4-D matrix with dimension _number * channel * width * height_ in HDF5 format. A good practice is to partition the dataset into small batches. 

#### Data manifest
For each of train/valid/test set,  a [train.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/train.txt)/[valid.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/valid.txt)/[test.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/test.txt) file is required to specify the ABSOLUTE PATH of all the HDF5 batches in the set.


## Caffe model file preparation
Refer to [here](http://caffe.berkeleyvision.org/) and [here](https://github.com/BVLC/caffe/tree/master/models) for instructions and examples in generating the following three files: 


+ [trainval.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/trainval.prototxt): architecture for training
+ [solver.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/solver.prototxt): solver parameters
+ [deploy.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/deploy.prototxt): architecture for testing

Note in trainval.prototxt

+ Data must be input through a hdf5 layer
+ Data source (for training as an example) should be either "../../../data/train.txt" or an absolute path $model_topdir$/data/train.txt.

## Prepare param.list
This file specificies all the parameters in the model. 

Example : (example/param.list)

```
model_topdir /$REPO_HOME$/example/
data_src NA
gpunum 0
solver_file example/solver.prototxt
trainval_file example/trainval.prototxt
deploy_file example/deploy.prototxt
codedir $REPO_HOME$
modelname basicmodel
model_batchname version0
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
Train a CNN using the parameters provided in `param.list`. $trial_num$ number of independent training will be performed using the same parameters. The output will be under $model_topdir$/$modelname$/$model_batchname$/


```
python run.py train param.list
```

#### Test
First, from all the model files saved across different trials and iterations, the one with best validation accuracy will be picked. Then input the test data into this model to generate prediction output. All the ouput in test phase are under /$model_topdir$/$modelname$/$model_batchname$/best_trial/

```
python run.py test param.list
```

#### Evaluate the testing result
Evaluate the performance on test set by accuracy and area under receiver operating curve (auROC). The output is in the same output folder as test phase.


```
python run.py test_eval param.list
```
