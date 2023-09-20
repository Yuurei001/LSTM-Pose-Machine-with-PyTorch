# LSTM Pose Machines for Video Human Pose Estimation - Implemented by PyTorch
This is a pytorch implementation of [LSTM Pose Machines 2017](https://arxiv.org/abs/1712.06316) 



## Overview:
This repository contains the PyTorch implementation of the research paper titled "LSTM Pose Machines" which introduces an approach for video-based human pose estimation. The model proposed in this paper utilizes convolutional LSTM units to capture temporal geometric consistency and dependencies among video frames for accurate pose estimation.



## Model
This network consists of T stages, where T is the number of frames. In each stage, one frame from a sequence will be sent into the network as input. ConvNet2 is a multi-layer CNN network for extracting features while an additional ConvNet1 will be used in the first stage for initialization. Results from the last stage will be concatenated with newly processed inputs plus a central Gaussian map, and they will be sent into the LSTM module. Outputs from LSTM will pass ConvNet3 and produce predictions for each frame. The architectures of those ConvNets are the same as the counterparts used in the [CPM model](https://arxiv.org/abs/1602.00134) but their weights are shared across stages. LSTM also enables weight sharing, which reduces the number of parameters in our network.

 <img src="https://github.com/HoseinAzad/LSTM-Pose-Machine-with-PyTorch/blob/master/ims/im1.png" width="1100" height="400" class="centerImage">
 

## Dataset
[Penn Action](http://dreamdragon.github.io/PennAction/) dataset is used to train the model. Penn Action Dataset (University of Pennsylvania) contains 2326 video sequences of 15 different actions and human joint annotations for each sequence. The dataset is available for download via the following link:
```
https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz
```


## References 
This code draws lessons from:<br>
https://github.com/HowieMa/lstm_pm_pytorch
