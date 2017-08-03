#**Behavioral Cloning** 

---

**Udacity Self-driving Car Nanodegree Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/center.jpg "Center Image"
[image2]: ./img/flipped.jpg "flipped Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 3 convolution layers of depths 16,24,32 (model.py lines 66,72,78), each followed by a max pooling layer (code line 69,75,81). 

####2. Process the input image

The image is cropped with a Keras Crop2D layer (code line 62). It is then normalized using a Keras lambda layer (code line 63).

####3. Using the right activation function

The model uses Leaky RELU as activation of convolutional layers (code line 67,73,79). The leaky alpha is set to 0.1 to give it a hint of leak (code line 18,65,71,77). 

####4. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code line 68,74,80). The dropout rate was set to 0.2 (code line 17).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####5. Optimizer

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

####6. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used  center lane driving, driving clock and driving multiply laps to obtain the data. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with different convolutional, activation and dropout layers.

My first step was to use a convolution neural network model similar to the VGG16 I thought this model might be appropriate because this is a very powerful network that learns about the features of images very well.

After experimenting transfer learning, I realized the task of behavioral cloning does not require such a powerful net as the lanes are merely lines and road single toned pixels.The LeNet transfer learning model simply took too long to train.

I then experimented with 2-4 convolution layers followed by max pooling layers. I made the depth of the layers small enough to train fast. I also experimented with the number of neurons in the last two fully connected layer and found that there should neither be too much nor too little.

With all these steps done, however, the car was not driving properly. It only knows to turn in one direction, and often it is the wrong direction. I thought that using only relu, the car ignored all the negative values in the nodes, resulting in bad results. So instead I used leaky relu, and the car immediately knows how to adjust itself both ways.

At this point, the car was able to go through the first turn successfully as well as over the bridge. However, the second turn lacked clear lanes on the right, and the car was not able to learn how to turn without it. This shows that the network was not robust enough. 

So I tuned the dropout rates from 0.8 down to 0.5, then all the way down to 0.2. By lowering the dropout rates, I was able to make the car depend only on 20% of the data for each layer. This allowed the car to drive even if there's not lane lines on the right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Input Shape = (None, 160, 320, 3)
_________________________________________________________________
1. Cropping2D | Output Shape = (None, 100, 320, 3)
_________________________________________________________________
2. Lambda
_________________________________________________________________
3. Conv2D | Output Shape = (None, 96, 316, 16)
_________________________________________________________________
....... LeakyReLU
_________________________________________________________________
....... Dropout     
_________________________________________________________________
....... MaxPooling2D | Output Shape = (None, 48, 158, 16)
_________________________________________________________________
4. Conv2D | Output Shape = (None, 44, 154, 24)
_________________________________________________________________
....... LeakyReLU
_________________________________________________________________
....... Dropout       
_________________________________________________________________
....... MaxPooling2D | Output Shape = (None, 22, 77, 24) 
_________________________________________________________________
5. Conv2D | Output Shape = (None, 18, 73, 32)
_________________________________________________________________
....... LeakyReLU
_________________________________________________________________
....... Dropout       
_________________________________________________________________
....... MaxPooling2D | Output Shape = (None, 9, 36, 32)     
_________________________________________________________________
..............Flatten | Output Shape = (None, 10368)        
_________________________________________________________________
6. Dense | Output Shape = (None, 512)
_________________________________________________________________
....... ReLU
_________________________________________________________________
....... Dropout
_________________________________________________________________
7. Dense | Output Shape = (None, 512)
_________________________________________________________________
....... ReLU
_________________________________________________________________
8. Dense | Output Shape = (None, 1)
_________________________________________________________________

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data sat, I flipped images and angles thinking that this would teach the car to turn the other direction. For example, here is an image that has then been flipped:

![alt text][image2]


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as the loss didn't drop after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
