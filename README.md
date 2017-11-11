# NYUDepthNet
NYUDepthNet is an unofficial Tensorflow implementation of *[Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture](https://www.cs.nyu.edu/~deigen/depth/)*. Note: This repository was created for a research project, not associated with NYU, to explore the implications of residual neural networks for monocular depth estimation and smartphone-based spatial mapping. These modifications were not included in this repository for compeleteness. If you would like these modification, please email me at rcbridendev@gmail.com. 


# Installation
There are two ways to install NYUDepthNet - Automatic and Manual. The latter is complex to configure, so it's recommended that you use the Automatic method.

## Automatic 
This is the recommended way to install NYUDepthNet.
* Clone the repository.
* Install dependencies
	* [Tensorflow](https://github.com/tensorflow/tensorflow)
	* [Numpy](https://github.com/numpy/numpy)
	* [Matplotlib](https://github.com/matplotlib/matplotlib)
* Run main.py to ensure NYUDepthNet was installed correctly.

## Manual
This installation method is more complex; however, it does grant increased customizability.
* Clone the repository.
* Install dependencies
	* [Theano](https://github.com/Theano/Theano)
	* Install the dependencies mentioned in the Automatic method.
* There are two methods to setup the installation:
	1. Run setup_env.py
	2. Manually download an unpack weights
		* Download weights and scripts from [NYU](https://cs.nyu.edu/~deigen/depth/)
		* Convert weights from .pk format to .npy or to tensorflow variables. __NOTE__: These weights are formatted as Theano tensors, so they must be converted to Tensorflow tensors. See [TheanoUnpickler](https://github.com/Rostifar/TheanoUnpickler).
* Run main.py to ensure NYUDepthNet was installed correctly.
