**Behavioral Cloning Project**
---
The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around tracks without leaving the road

[//]: # (Image References)

[image1]: ./Images/SampleData.png "Sample Data Steering Angles"
[image2]: ./Images/ScaledSampleData.png "Sample Data Steering Angles scaled by a factor of 0.5"
[image3]: ./Images/AugSampleData.png "Augmented Data Steering Angles"
[image4]: ./Images/FlippedImage.png "Flipped Image and Steering Angle"
[image5]: ./Images/Track2Data.png "Scaled & Augmented Track 2 Steering Angle Data"
[image6]: ./Images/Track2Cropped.png "Original & Cropped Image from Track 2"
[image7]: ./Images/Track2Resized.png "Resized image from Track 2"

## 1. Project Files

The following files are available in this repository:
* [model.py](https://github.com/shazraz/Behavioral-Cloning/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/shazraz/Behavioral-Cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/shazraz/Behavioral-Cloning/blob/master/model.h5) containing the trained convolution neural network 

## 2. Executing the model
The car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
in the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim). The model is capable of driving the vehicle around both Track 1 and Track 2 in autonomous mode with the simulator set to the "Fastest" graphics setting.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and contains comments to explain how the code works.

## 3. Model Architecture and Training Strategy

**3.1 Model Architecture**

The model is built using Keras Sequential model API with TensorFlow on the backend. The model consists of the following layers as reported by model.summary():

| Layer (type)            | Output Shape        | Param # | Comments                                                                                                       |
|-------------------------|---------------------|---------|----------------------------------------------------------------------------------------------------------------|
| Cropping (Cropping2D)   | (None, 45, 100, 5)  | 0       | Input shape of 100x100 pixels cropped to 45x100 pixels by removing 40 rows from the top and 15 from the bottom |
| Normalization (Lambda)  | (None, 45, 100, 5)  | 0       | Scales pixel values by 255 and subtracts 0.5 to normalize pixel values between -0.5 and 0.5                    |
| Conv1 (Convolution2D)   | (None, 45, 100, 12) | 552     | 3x3 kernel size with a stride of 1 and SAME padding. Normal distribution initialization of parameters with an ELU activation function.                                                            |
| MaxPool1 (MaxPooling2D) | (None, 22, 50, 12)  | 0       | 2x2 kernel size                                                                                                |
| Conv2 (Convolution2D)   | (None, 22, 50, 36)  | 3924    | 3x3 kernel size with a stride of 1 and SAME padding. Normal distribution initialization of parameters with an ELU activation function.                                                            |
| MaxPool2 (MaxPooling2D) | (None, 11, 25, 36)  | 0       | 2x2 kernel size                                                                                                |
| Conv3 (Convolution2D)   | (None, 11, 25, 48)  | 15600   | 3x3 kernel size with a stride of 1 and SAME padding. Normal distribution initialization of parameters with an ELU activation function.                                                            |
| MaxPool3 (MaxPooling2D) | (None, 5, 12, 48)   | 0       | 2x2 kernel size                                                                                                |
| Conv4 (Convolution2D)   | (None, 5, 12, 96)   | 41568   | 3x3 kernel size with a stride of 1 and SAME padding. Normal distribution initialization of parameters with an ELU activation function.                                                            |
| MaxPool4 (MaxPooling2D) | (None, 2, 6, 96)    | 0       | 2x2 kernel size                                                                                                |
| Flatten (Flatten)       | (None, 1152)        | 0       | Flattening layer                                                                                               |
| FC1 (Dense)             | (None, 1280)        | 1475840 | Fully connected layer with ELU activation function                                                                                         |
| Dropout1 (Dropout)      | (None, 1280)        | 0       | Dropout to reduce overfitting                                                                                  |
| FC2 (Dense)             | (None, 320)         | 409920  | Fully connected layer with ELU activation function                                                                                          |
| Dropout2 (Dropout)      | (None, 320)         | 0       | Dropout to reduce overfitting                                                                                  |
| FC3 (Dense)             | (None, 80)          | 25680   | Fully connected layer with ELU activation function                                                                                          |
| Dropout3 (Dropout)      | (None, 80)          | 0       | Dropout to reduce overfitting                                                                                  |
| Output (Dense)          | (None, 1)           | 81      | Output layer                                                                                                   |

This results in a total trainable parameters of 1,973,165

**3.2 Addressing Overfitting**

The model contains three dropout layers applied to the outputs Dense layers FC1, FC2 and FC3 to combat overfitting. L2 regularization was also experimented with but not found to be necessary and degraded the performance of the model with even very low beta values (<0.0001)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 269-271). An early stopping callback was also applied to the model with a delta of 0.0075 on the validation loss and a patience of 1 epoch to stop further training of the model if the validation loss did not monotonously decrease. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

**3.3 Parameter Tuning**

The model used an adam optimizer to minimize the MSE loss, so the learning rate was not tuned manually (model.py line 265). However, additional hyper parameters were introduced into the model during the course of development which will be explained in later sections:

* Batch_Size: size of batch to pass to generators
* Keep_Prob: probability of keeping parameters in dropout layer
* Scale_Factor: factor to scale the zero angle driving data to balance the training data set
* Flip_Angle: steering angle threshold over which images are flipped for data augmentation 

**3.4 Gathering Training Data**

Training data was initally gathered on Track 1 using the Udacity provided recommendations. i.e. center lane driving in both directions of the track to generate a balanced data set with some "recovery" driving to help the model recover the vehicle from unexpected situations. However, the final training data set used to train the model consisted of 3 laps of driving data collected from only the challenge track (Track 2). The collected data also included instances of recovery driving as they naturally occured while manually driving the tracks. Details of the problem exploration and justification for the training data and design steps are discussed in the following section.

## 4. Solution Design

**4.1 Initial Model**

The sample data provided with the project was visualized to determine the distribution of the steering angles collected. This is visualized in the image below.

![alt text][image1]

The available data is highly biased towards driving straight with an excessive amount of data points between less than |0.05| (where the steering angle is scaled to between -1 and 1). This was addressed by scaling the data points within this angle range by a factor of Scale_Factor, a hyper-parameter available for tuning. The scaled steering angles are shown below scaled by a factor of 0.5: 

![alt text][image2]

The dataset can further be balanced by flipping the images and steering angles for images where the steering angle is greater than a threshold of Flip_angle, where this parameter can also be tuned to augment the data. The augmented dataset for the sample data is visualized below. A threshold value of 0.05 was used.

![alt text][image3]

The images below show an image collected from Track 1 and a flipped image with the steering angle multiplied by -1.

![alt text][image4]

This dataset was then used to train an initial model consisting of 3 convolutional layers and 3 fully connected layers. The input RGB image had 35 of the top pixel rows and 10 of the bottom pixel rows cropped and was resized to 80x80. The convolutional layers were identical to the layers defined in the final model above whereas the three fully connected layers after the Flatten layer had 640, 460 and 40 parameters respectively. The dataset was split into a training split (80%) and validation split (20%). A Keep_Prob of 0.5 was used on the training data and the model was trained for 5 epochs. 

This model was able to navigate the vehicle the majority of the way around Track 1 but was unable to do the sharper right turn encountered after the bridge in the first track. Two laps of additional data were collected on Track 1 driving the oppsite way around the track to provide additional training data. The model trained on the sample data was loaded (using the -m argument in model.py) and trained on the additional data collected using a scale factor of 0.3 and Flip_angle of 0.05 for 3 epochs. 

This amount of data augmentation and training was sufficient to navigate the vehicle indefinitely around Track 1. This can be seen in the video linked [here](https://youtu.be/3Y8kd1PHrOk). 

However, this model generalized very poorly to the challenge track where the vehicle ran into the barrier almost immediately after starting. Therefore, a more generalized model was required that would be able to navigate both tracks. 

**4.2 Gathering More Training Data**

The visualizations above show that the steering angle distribution available for Track 1 has a relatively Gaussian distribution and can therefore only perform soft right and left turns. As a result, a model trained on this training data would be unable to perform the hard right or left turns that require higher steering angles to successfully navigate the challenge track. This lack of steering data for sharper right/left turns could be obtained by using the right and left camera images on Track 1 and adding an offset to their steering angles, however, it was decided to collect data on Track 2 instead and generalize the model enough so that it would successfully drive Track 1 as well.

The visualization below shows the steering angle distribution for 3 laps of data collected from Track 2. While collecting the training data, the vehicle was driven over the lane striping between the two lanes. Any "recovery" driving that naturally occured during the course of driving the 3 laps was also collected. The visualization below shows the data distribution scaled by a factor of 0.5 and augmented by flipping images and angles over the 0.05 angle threshold.

![alt text][image5]

The image above shows a much more balanced dataset and a model trained on this data would be capable of navigating much sharper turns. However, Track 2 contains a large number of uphill and downhill segments compared to Track 1 so the cropping of the images was adjusted to account for this. The top of the image was cropped by 40 pixel rows compared to 35 and the bottom of the image by 15 pixel rows compared to 10 to provide a narrower view of the scene to the model during training. The resulting image was resized to 100x100 as opposed to the 80x80 used for the initial model. The Track 2 images below show the original image, cropped image and resized images used to train the model.

![alt text][image6]

![alt text][image7]

**4.3 Model Training & Generalization**

The initial model described above was then trained on the Track 2 training data visualized in the previous section. 


The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
