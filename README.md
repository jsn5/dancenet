# DanceNet - Dance generator using Variational Autoencoder, LSTM and Mixture Density Network. (Keras)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jsn5/dancenet/blob/master/LICENSE) [![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)
[![DOI](https://zenodo.org/badge/143685321.svg)](https://zenodo.org/badge/latestdoi/143685321)


This is an attempt to create a dance generator AI, inspired by [this](https://www.youtube.com/watch?v=Sc7RiNgHHaE&t=9s) video by [@carykh](https://twitter.com/realCarykh)


![](https://github.com/jsn5/dancenet/blob/master/demo.gif ) ![](https://github.com/jsn5/dancenet/blob/master/demo2.gif )

## Main components:

* Variational autoencoder
* LSTM + Mixture Density Layer

## Requirements:

* Python version = 3.5.2

  ### Packages
  * keras==2.2.0
  * sklearn==0.19.1
  * numpy==1.14.3
  * opencv-python==3.4.1

## Dataset

https://www.youtube.com/watch?v=NdSqAAT28v0
This is the video used for training.


## How to run locally

* Download the trained weights from [here](https://drive.google.com/file/d/1LWtERyPAzYeZjL816gBoLyQdC2MDK961/view?usp=sharing). and extract it to the dancenet dir.
* Run dancegen.ipynb

## How to run in your browser

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run)

* Click the button above to open this code in a FloydHub workspace (the [trained weights dataset](https://www.floydhub.com/whatrocks/datasets/dancenet-weights) will be automatically attached to the environment)
* Run dancegen.ipynb

## Training from scratch

* fill dance sequence images labeled as `1.jpg`, `2.jpg` ... in `imgs/` folder
* run `model.py`
* run `gen_lv.py` to encode images
* run `video_from_lv.py` to test decoded video
* run  jupyter notebook `dancegen.ipynb` to train dancenet and generate new video.

## References

* [Original project video](https://www.youtube.com/watch?v=Sc7RiNgHHaE&t=9s) by Cary Huang
* [Building autoencoders in keras](https://blog.keras.io/building-autoencoders-in-keras.html) by [Francois Chollet](https://twitter.com/fchollet)
* [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
* [Generative Choreography using Deep Learning](https://arxiv.org/abs/1605.06921)
* [Mixture Density Networks](http://blog.otoro.net/2015/06/14/mixture-density-networks/) by [David Ha](https://twitter.com/hardmaru)
* [Mixture Density Layer for Keras](https://github.com/cpmpercussion/keras-mdn-layer) by [Charles Martin](https://github.com/cpmpercussion/)
 
