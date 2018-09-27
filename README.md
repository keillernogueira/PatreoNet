# PatreoNet

The performance of image classification is highly dependent on the quality of extracted features.
Concerning high-resolution remote sensing image images, encoding the spatial features in an efficient and robust fashion is the key to generating discriminatory models to classify them.
Even though many visual descriptors have been proposed or successfully used to encode spatial features of remote sensing images, some applications, using this sort of images, demand more specific description techniques.
Deep Learning, an emergent machine learning approach based on neural networks, is capable of learning specific features and classifiers at the same time and adjust at each step, in real time, to better fit the need of each problem.
In this work, we introduced and evaluated the benefits of deep learning (specifically convolutional network) into the remote sensing domain.
Specifically, a new network, called PatreoNet, is proposed to perform remote sensing image classification.
The network, that has six layers (three convolutional, two fully-connected and one classifier layer), was tested in two remote sensing datasets:

  - the popular aerial image dataset [UCMerced Land-use](http://vision.ucmerced.edu/datasets/landuse.html)
  - a multispectral high-resolution scenes of the [Brazilian Coffee Scenes](http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-coffee-scenes-dataset/)
  
## Reimplementation

This repository is a reimplementation of the proposed PatreoNet.
Originally implemented using [Caffe](http://caffe.berkeleyvision.org/) framework, here, the network was implemented using [TensorFlow](http://tensorflow.org/) (0.10.0).

	
## Reimplementation

The trained model for the UCMerced Land-use can be found [here](https://www.dropbox.com/s/nxddpnpij1yqexo/PatreoNet_UCMerced_model.zip?dl=0).

  
## Citing

If you use this code in your research, please consider citing:

    @inproceedings{nogueira2015improving,
	  title={Improving spatial feature representation from aerial scenes by using convolutional networks},
	  author={Nogueira, Keiller and Miranda, Waner O and Dos Santos, Jefersson A},
	  booktitle={Graphics, Patterns and Images (SIBGRAPI), 2015 28th SIBGRAPI Conference on},
	  pages={289--296},
	  year={2015},
	  organization={IEEE}
	}
