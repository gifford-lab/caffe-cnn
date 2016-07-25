A platform for training and testing convolutional neural network (CNN) based on [Caffe](http://caffe.berkeleyvision.org/)

## Prerequisite
[Caffe and pycaffe](http://caffe.berkeleyvision.org/installation.html) (already included for mri-wrapper)

## Quick run
We provide some toy data for you to quickly play around.

+ First replace all $REPO_HOME$ in the following files as the actual directory of the repository:

	+ example/data/train.txt
	+ example/data/valid.txt
	+ example/data/test.txt


+ Train a simple neural network on the toy training data and find the model snapshot with the best performance on validation set :
	
	```
python run.py train example/param.list
```

+ Make prediction on the toy test set:

	```
python run.py test example/param.list
```

+ Evaluate the prediction:

	```
python run.py test_eval example/param.list
```

The output of the above three commands will be under *$REPO_HOME$/example/basicmodel/version0/*


+ Predict on new data with the trained network:

	```
python run.py predict example/param.list
```

The output will be *$REPO_HOME$/example/newdata_output*
	

For more details on how to run the program. scroll down to the 'Ready to go' section.

## Data preparation


#### For DNA Sequence data
+ Prepare your sequence of training, validation and testing set separately in tab-separated values ([TSV](https://en.wikipedia.org/wiki/Tab-separated_values)) format([Example](https://github.com/gifford-lab/caffe-cnn/tree/master/example/sequence/sample.tsv))
+ Prepare the target (the label or real value you want to fit) for your training, validation and testing set separately. One line for each sample.
+ Use this [script](https://github.com/gifford-lab/caffe-cnn/tree/master/embedH5.py) to transform the sequence and target to the data format that Caffe can take. 
	+ Usage example:
	
		```
		python embedH5.py tsvfile targetfile outfile -p $output_dir$
		```
	+ __Mri-wrapper users shouldn't include the "-p" argument.__
	
	+ The "outfile" argument for training, validation and testing set should be the following respectively:
 
 		```
 $output_topdir$/data/train.h5
 $output_topdir$/data/valid.h5
 $output_topdir$/data/test.h5
 ```
	
	
 	+ Type the following for details on other optional arguments:
 
		```
 python embedH5.py -h
 	```
 	
#### For other data type
You will need to manually prepare the following data under *$output_topdir$/data/*

+ Data matirx

	Training, validating and testing data should be each prepared as 4-D matrix with dimension _number * channel * width * height_ in HDF5 format. A good practice is to partition the dataset into small batches. ([Example](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data))

+ Data manifest

	For each of train/valid/test set,  a [train.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/train.txt)/[valid.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/valid.txt)/[test.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/test.txt) file is required to specify the ABSOLUTE PATH of all the HDF5 batches in the set.
	__Mri-wrapper users should set the top folder as "/data" so that for example each line in [train.txt](https://github.com/gifford-lab/caffe-cnn/tree/master/example/data/train.txt) should be /data/data/train.h5.batchX__


## Caffe model file preparation
Refer to [here](http://caffe.berkeleyvision.org/) and [here](https://github.com/BVLC/caffe/tree/master/models) for instructions and examples in generating the following three files: 


+ [trainval.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/trainval.prototxt): architecture for training
+ [solver.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/solver.prototxt): solver parameters
+ [deploy.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/deploy.prototxt): architecture for testing

Note in trainval.prototxt

+ Data must be input through a hdf5 layer
+ Data source (for training as an example) should be either *../../../data/train.txt* or an absolute path *$output_topdir$/data/train.txt*

## Prepare [param.list](https://github.com/gifford-lab/caffe-cnn/blob/master/example/param.list)

Prepare a space-delimited file that specifies what you wish to do.

General argument:
+ `gpunum`: The GPU device number to run on

If you want to perform training, testing and performance evaluation:
+ `output_topdir`: The top output folder for training and testing.
+ `caffemodel_topdir`: The directory where you put all the caffe model files
+ `model_name`: The name of the model. 
+ `batch_name`: The name of the specific model version. All the output will be therefore generated under *$output_topdir$/$modelname$/$model_batchname$/*
+ `batch2predict`: The model version to make prediction with.
+ `trial_num`: The number of training trial.
+ `optimwrt`: choose the best trial wrt this metric (accuracy or loss)
+ `outputlayer`: The name of the layer of which the output you wish to save when you apply the model on the test set. Different layers should be separated by "_" for example "prob_fc2". 

If you want to  predict on new data with a trained model:
+ `outputlayer`: Same as above.
+ `deploy2predictW`: path to the [deploy.prototxt](https://github.com/gifford-lab/caffe-cnn/blob/master/example/deploy.prototxt) file of the trained model, which specifies the network architecture.
+ `caffemodel2predictW`: path to the snapshot of the trained model. See 'train a neural network' section below about where the best model parameters are saved after training.
+ `data2predict`: path to the [manifest](https://github.com/gifford-lab/caffe-cnn/blob/master/example/data/test.txt) file specifying all HDF5 format data to predict on.
+ `predict_outdir`:  the output directory, under which 'bestiter.pred' and 'bestiter.pred.params.pkl' will be saved. See description for `outputlayer`.

## Ready to go!


#### Train a neural network
$trial_num$ number of independent training will be performed using the same parameters. The output will be under *$output_topdir$/$model_name$/$batch_name$/*. File 'train.err' saves the log for each trial. Then from all the model files saved across different trials and iterations, the one with best validation performance (evaluated by 'optimwrt') will be picked and saved as *bestiter.caffemodel* and *deploy.prototxt* under *$output_topdir$/$model_name$/$batch_name$/best_trial/*.


```
python run.py train param.list
```

#### Make prediction with the trained neural network
The best model picked above will be used to predict on the test set.  The output of the first layer specified in `outputlayer` (see above section on params.list) will be saved to a text file named *bestiter.pred* and the output of all layers will be saved in a Pickle object named *bestiter.pred.params.pkl*. They are both under *$output_topdir$/$model_name$/$batch_name$/best_trial/*.

```
python run.py test param.list
```

#### Evaluate the prediction performance
Evaluate the performance on test set by accuracy and area under receiver operating curve (auROC). It takes *bestiter.pred* as input and the output is *bestiter.pred.eval* in the same folder.

```
python run.py test_eval param.list
```

#### Predict with a trained model
In param.list, the user has to specify the structure (deploy.prototxt) and parameter (*.caffemodel) files for the trained model, the data manifest to predict on, and the output directory. See 'Prepare param.list' section for more details.

````
python run.py predict param.list 
````
