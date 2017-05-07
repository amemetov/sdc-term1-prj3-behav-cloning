# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center]: ./report/center.jpg "center lane driving"
[recovery1]: ./report/recovery1.jpg "Recovery 1"
[recovery2]: ./report/recovery2.jpg "Recovery 2"
[recovery3]: ./report/recovery3.jpg "Recovery 3"
[recovery4]: ./report/recovery4.jpg "Recovery 4"
[recovery5]: ./report/recovery5.jpg "Recovery 5"
[right-side]: ./report/right-side.jpg "Origin"
[right-side-flipped]: ./report/right-side-flipped.jpg "Flipped"
[right-side-cropped]: ./report/right-side-cropped.jpg "Cropped"
[right-side-brightness1]: ./report/right-side-brightness1.jpg "Brightness 1"
[right-side-brightness2]: ./report/right-side-brightness2.jpg "Brightness 2"
[right-side-saturation1]: ./report/right-side-saturation1.jpg "Saturation 1"
[right-side-saturation2]: ./report/right-side-saturation2.jpg "Saturation 2"
[loss-curve]: ./loss-curve.png "Train/Valid Loss History Curves"
[loss-curve2]: ./loss-curve2.png "Model2 Train/Valid Loss History Curves"

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [utils.py](utils.py) containing helper methods loading data, generating data batches, flipping images, etc.
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used [NVidia's solution](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 128 ([model.py](model.py), method nvidia_model, lines 18-57) 

The model includes RELU layers to introduce nonlinearity (code line 66), and the data is normalized in the model using a Keras lambda layer (code line 22).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ([model.py](model.py) lines 72 and 80). 

The model was trained and validated on different data sets to ensure that the model was not overfitting ([utils.py](utils.py) code line 287, 293, 294). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer the default learning rate (0.001) ([model.py](model.py) line 165).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

I tried 2 models:
* 1. My model from [Traffic Sign Recognition Project](https://github.com/amemetov/sdc-term1-prj2-traffic-sign-classifier).
* 2. [NVidia model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

First of all I split data set into a training (90%) and validation set (10%) ([utils.py](utils.py) line 287).

My model from Traffic Sign Project ([model.py](model.py) lines 201-226) produced low mean squared error both for train and validation sets, 
but produced poor result in the simulator - the vehicle could not stay on the track.

Then I have implemented NVidia model ([model.py](model.py), method nvidia_model, lines 18-57). 
During traning I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. To combat the overfitting, I modified the model so that it used Dropout.
By using GridSearch I found out that dropout_prob=0.2 is a good value for this model and dataset.
It allowed to get rid off overfitting.
For non linearity I have tried ReLU and ELU and did not find a difference in the result (neither for convergence speed nor in the simulator).
I have tried BatchNormalization too - it did not give any improvement (convergence speed was the same, mse values were similar, no improvements in the simulator).
Also I gave a try to different kernel sizes, strides, poolings for CONV layers - there were no improvements at all.

The most improvement was done by getting more and good/right data from the simulator.
There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I have just driven there several times,
but got new samples did not improve the result.
After some investigation of the distribution of steering  values for collected data I understood that using keyboard does not allow to get smoothed steering values
(as for example it would be if I drive on the real car using steering wheel).
Then I found out that I can use a mouse, after some training I was able to use mouse to get more realistic steering values.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture ([model.py](model.py), method nvidia_model, lines 18-57) 
consisted of a convolution neural network with the following layers and layer sizes:
| Layer | Size |
| Normalization | (66, 200, 3) | 
| Convolution2D - ReLU - Dropout | (31, 98, 24) |
| Convolution2D - ReLU - Dropout | (14, 47, 36) |
| Convolution2D - ReLU - Dropout | (5, 22, 48) |
| Convolution2D - ReLU - Dropout | (3, 20, 64) |
| Convolution2D - ReLU - Dropout | (1, 18, 64) |
| Flatten | 1152 |
| FC - ReLU - Dropout | 1164 |
| FC - ReLU - Dropout | 100 |
| FC - ReLU - Dropout | 50 |
| FC - ReLU - Dropout | 10 |
| FC | 1 |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do when it’s off on the side of the road.

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]
![alt text][recovery4]
![alt text][recovery5]

Then I repeated this process on track two in order to get more data points.

The simulator produces images of size 160x320 pixels, but not all pixels contain usefull information.
I cropped 40 pixels from the top of the image and 20 pixels from the bottom.

![alt text][right-side]
![alt text][right-side-cropped]


To augment the dataset, I also flipped images and angles thinking that this would help with the left turn bias.
For example, here is an image that has then been flipped:

![alt text][right-side]
![alt text][right-side-flipped]


I also used randomization of brightness and saturation to help the model to get more generalisation:
![alt text][right-side]
![alt text][right-side-brightness1]
![alt text][right-side-brightness2]
![alt text][right-side-saturation1]
![alt text][right-side-saturation2]


I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
I used EarlyStopping ([utils.py](utils.py) line 299) to stop training when validation mse has stopped improving.
I used an adam optimizer with the default learning rate 0.001 ([model.py](model.py) line 165).

The plot of train/valid loss history curves:
![alt text][loss-curve]

The result is presented in below video files:
* [Track1](model1-track1.mp4)
* [Track2](model1-track1.mp4)
* [Track1 on Youtube](https://www.youtube.com/watch?v=nvXf9Y3PsfQ)
* [Track2 on Youtube](https://www.youtube.com/watch?v=FUoYBcHUC2w)




### Extra Model
Due testing the model on the track2 I noticed that I need to reduce the desired speed to get car on the track (15 mph comparing to 20 mph for track1).
I decided that it would be better if the model will predict not only the steering but the speed too (instead of hardcoding the speed value in the code).
So I have implemented a model (I called it model2) which predicts 2 values: steering and speed.
Files for model2:
* [model2.py](model2.py)
* [drive2.py](drive2.py)

To drive the car using model2 run:
```sh
python drive2.py model2.h5
```

The model is the same NVidia model, but with 2 outs.

The result is presented in below video files:
* [Model2 - Track1](model2-track1.mp4)
* [Model2 - Track2](model2-track1.mp4)
* [Model2 - Track1 on Youtube](https://www.youtube.com/watch?v=ym1SOPAOyo8)
* [Model2 - Track2 on Youtube](https://www.youtube.com/watch?v=E6M2Y4eVz_c)

I think that this approach gives more natural driving style cause it for example decreases speed at complex spots (dangerous curve) and increases speed on the straight road.

At the first approach I tried to build a model which will predict throttle and brake depending on the current image and the current speed.
For that I created a [model3](model3.py) with 2 inputs (image and speed), merge layer and 2 outputs (steering and throttle). 
Throttle and brake was combined (just summed up) cause we cannot pass brake to the simulator.
But I did not manage to find a good working solution, so it is my the first goal after finishing Term1. 








