# DanceNet - Dance generator using Autoencoder, LSTM and Mixture Density Network. (Keras)
[![Open Source Love](https://badges.frapsoft.com/os/mit/mit.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=102)](https://github.com/jsn5/dancenet/)

This is an attempt to create a dance generator AI, inspired by [this](https://www.youtube.com/watch?v=Sc7RiNgHHaE&t=9s) video by [@carykh](https://twitter.com/realCarykh)


![](https://github.com/jsn5/dancenet/blob/master/demo.gif )

## Main components:

* Variational autoencoder
* LSTM + Mixture Density Layer

## Requirements:

* keras==2.2.0
* sklearn==0.19.1
* numpy==1.14.3
* opencv-python==3.4.1

## Dataset

https://www.youtube.com/watch?v=NdSqAAT28v0
This is the video used for training.


## How to run

* Download the trained weights from [here](https://drive.google.com/file/d/1LWtERyPAzYeZjL816gBoLyQdC2MDK961/view?usp=sharing). and extract it to the dancenet dir.
* Run dancegen.ipynb


## References

* [Original project video](https://www.youtube.com/watch?v=Sc7RiNgHHaE&t=9s) by Cary Huang
* [Building autoencoders in keras](https://blog.keras.io/building-autoencoders-in-keras.html) by [Francois Chollet](https://twitter.com/fchollet)
* [Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
* [Mixture Density Networks](http://blog.otoro.net/2015/06/14/mixture-density-networks/) by [David Ha](https://twitter.com/hardmaru)
* [Mixture Density Layer for Keras](https://github.com/cpmpercussion/keras-mdn-layer) by [Charles Martin](https://github.com/cpmpercussion/)
 
