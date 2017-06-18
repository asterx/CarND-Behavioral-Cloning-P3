#**Behavioral Cloning**


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[net]: ./images/net.png "Model Visualization"
[center_lane_driving]: ./images/center_lane_driving.gif "Center lane driving"
[turning]: ./images/turning.gif "Turning"
[recovery]: ./images/recovery.gif "Steering back to the middle of the road"
[original]: ./images/original.png "Original image"
[cut_and_resize]: ./images/cut_and_resize.png "Image after cutting top, bottom and applying resize"
[shadow]: ./images/shadow.png "Shadow"
[flip]: ./images/flip.png "Random flip"



## Rubric Points
My implementation inspired by great work of Mariusz Bojarski, Davide Del Testa and others (original paper could be found [here](https://arxiv.org/abs/1604.07316)). Model described in this submission works with smaller images and contains less layers but it is still shows advantages of chosen approach.

---
###Files Submitted & Code Quality

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [jupiter notebook](/CarND-Behavioral-Cloning.ipynb) with code used to train the model (I run my code on remote server and it's very useful to have web interface; on the late stages of development, model implementation and various utils were moved to separate files); same code could be found at [train.py](/train.py)
* [net/net.py](/net/net.py) containing code that describes my model
* [drive.py](/drive.py) for driving the car in autonomous mode (code changed from original I added ablility to load JSON with model and weights)
* [model/](/model) containing a trained convolution neural network
* writeup_report.md summarizing the results


#### Running jupiter notebook
```sh
./run.sh
```

#### Running model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model/model.json model/epoch_19.h5
```

#### Appropriate training data
It was clear from the beginning that simple driving on the first track won't be enough, so I also tried different techniques like driving correction - letting car drive off the
center and then steer it back to the middle of the road. Also 1st track has a lot of places where car should turn left and very few where it should turn right.
So I made data transition flipped left and right and corresponding steering values to obtain more various data at the end.
I manually collected several hours of training data on both tracks and then joined it with data provided by Udacity.
Obviously all these data still not enough for good training, so I applied some augmentation techniques (described in sections below).



### Model Architecture and Training Strategy

#### Solution Design Approach
As I mentioned earlier, architecture of my model inspired by model described in NVIDIA paper.
Original model was too big and complicated in terms of computing costs, so I simplified it by reducing number of 
 convolutional layers; Also I changed kernel size of all convolutional layers to 3.
To combat over fitting I decided to use dropout [50% on fully-connected layers](/net/net.py#L18-L20).
The model was trained and validated on different data sets to ensure that the model was not over fitting (see [train.py line 15](/train.py#L15) and [train.py lines 23-26](/train.py#L23-L26)).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

After hours of training, my model could successfully drive car in simulator.
There are no problems at all with 1st track, but there were a few spots where the vehicle fell off the track.
To improve that behavior I added more training data (originally training data did not include driving correction examples).
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Videos could be found on youtube:

[![video of track 1](https://img.youtube.com/vi/m1p26BkRnF8/0.jpg)](https://youtu.be/m1p26BkRnF8)
[![video of track 2](https://img.youtube.com/vi/YH2kp8y3ugU/0.jpg)](https://youtu.be/YH2kp8y3ugU)

#### Final Model Architecture
After several attempts I came up with following architecture (basically cut some convolutional layers):

<img width="300" src="https://github.com/asterx/CarND-Behavioral-Cloning-P3/blob/master/images/net.png">



#### Creation of the Training Set & Training Process
Firstly I captured good driver behavior - recorded several laps on the track using only center lane driving.
Then I spend some time driving more optimal on on the turns (example: left turn means that we should move close to the left side of the road).
After that I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep itself in the middle of the road.

![Center lane driving][center_lane_driving]
![Turning][turning]
![Steering back to the middle of the road][recovery]

After that I spend some time on the second track to get more data points.
As was mentioned before, some augmentation techniques were applied to collected data:

0. Select image from random camera
1. Cutting top and bottom
2. Resizing result image to 100x33
3. Adding random shadows
4. Random flipping

Image transformation pipeline illustrated:

![Original image][original]
![Cut and resize][cut_and_resize]
![Shadow][shadow]
![Flip][flip]

After the collection process, I had ~ 20 000 of data points. Obviously data set was very unbalanced and had many examples with steering positions close to 0.
I decided to apply designated random sampling to make sure that data is balanced across steering angles (basically this means that splitting steering angles were split inti 1024 bins, with 512 frames for each bin; see [utils/data.py lines 21-30](/utils/data.py#L21-L30) for more details).

After all preparations data was randomly shuffled and put 20% of the data was into a validation set.
The validation set helped determine if the model was over or under fitting.
The ideal number of epochs was 20 (accuracy and loss values stopped improving significantly).
I used an Adam optimizer so that manually training the learning rate wasn't necessary.
