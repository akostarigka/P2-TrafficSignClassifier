## Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_OrigVsGrayVsNorm]: ./writeup_figures/OrigVsGrayVsNorm.png "Original vs. Grayscale vs. Normalized Grayscale"
[image_trainingdataset]: ./writeup_figures/TrainingDataSet.png "Training Dataset"
[image_validationdataset]: ./writeup_figures/ValidationDataSet.png "Validation Dataset"
[image_AugmentedTrainingSet]: ./writeup_figures/AugmentedTrainingSet.png "Augmented Training Set"
[image_trans_original]: ./writeup_figures/TransOriginal.png "Original Image"
[image_trans]: ./writeup_figures/Trans1.png "Transformed Images"
[image_TestImages]: ./writeup_figures/AllTestImages.png "Test Images"
[image_ts_test1]: ./writeup_figures/3_SpeedLimit60Kmh.jpg "Speed Limit 60Kmh"
[image_ts_test2]: ./writeup_figures/11_RightOfWay.jpg "Right Of Way"
[image_ts_test3]: ./writeup_figures/14_Stop.jpg "Stop"
[image_ts_test4]: ./writeup_figures/17_NoEntry.jpg "No Entry"
[image_ts_test5]: ./writeup_figures/18_GeneralCaution.jpg "General Caution"
[image_ts_test6]: ./writeup_figures/22_BumpyRoad.jpg "Bumpy Road"
[image_ts_test7]: ./writeup_figures/25_RoadWork.jpg "Road Work"
[image_ts_test8]: ./writeup_figures/28_ChildrenCrossing.jpg "Children Crossing"
[image_ts_test9]: ./writeup_figures/38_KeepRight.jpg "Keep Right"
[image_pred_1]: ./writeup_figures/Pred_60Kmh.png "Predictions for Speed Limit 60Kmh"
[image_pred_2]: ./writeup_figures/PredRoadWork.png "Predictions for Road Work"
[image_pred_3]: ./writeup_figures/PredChildrenCr.png "Predictions for Children Crossing"
[image_pred_4]: ./writeup_figures/PredKeepRight.png "Predictions for Keep Right"
[image_pred_5]: ./writeup_figures/PredStop.png "Predictions for Stop"
[image_pred_6]: ./writeup_figures/PredGeneralCaution.png "Predictions for General Caution"
[image_pred_7]: ./writeup_figures/PredBumpyRoad.png "Predictions for Bumpy Road"
[image_pred_8]: ./writeup_figures/PredRightOffWay.png "Predictions for Right Off Way"
[image_pred_9]: ./writeup_figures/PredNoEntry.png "Predictions for No Entry"

### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

The summary statistics of the traffic signs data set are calculated using the Numpy library as follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

As exploratory visualizations, the following bar charts show the distribution of the various labels in the training and validation datasets. One can notice that the number of samples varies in each label. This may have to do with the frequency each traffic sign is met in the real world. However the distribution of the labels is more or less kept in both sets (training, validation).   

##### Training Dataset
![alt text][image_trainingdataset]
##### Validation Dataset
![alt text][image_validationdataset]

### Model Architecture Design and Testing

#### 1. Image data preprocessing

##### Conversion to grayscale

Initially the images are converted to grayscale. The main reason why grayscale representations are often used in image recognition instead of color images is that grayscale simplifies the algorithm and reduces computational requirements. Color may be of limited benefit in many applications and introducing unnecessary information could increase the amount of training data required to achieve good performance.

##### Image data normalization

It has been proved that an adequate normalization previous to the training process is very important to obtain good results and to reduce significantly calculation time. Therefore, as a next step, image data were normalized into having zero mean and equal variance.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image_OrigVsGrayVsNorm]

##### Data augmentation

In order to boost the performance of the neural network, additional data were generated. It is known that the more data a classification algorithm has access to, the more effective it can be. Even when the data is of lower quality, algorithms can actually perform better, as long as useful data can be extracted by the model from the original data set [Jason Wang & Luis Perez, *The Effectiveness of Data Augmentation in Image Classification using Deep Learning*].

To add more data to the the data set, the function proposed by [Vivek Yadav](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3) was used, which generates transformed images having varying characteristics concerning their rotation, shift, brightness and shearing.

Here is an example of the 10 transformed images:

![alt text][image_trans]

Noticing the big difference in the amount of data in each unique label (Section 2), I decided to augment each one with a different number of samples. Like this we would have a more homogenous distribution of samples for each label. The formula to retrieve these augmentation numbers was: *(maximum_number_of_counts/label_counts)/1.5*

#### 2. Model architecture

The model used is based on the well known LeNet architecture with the following adaptations: two extra convolutions were added and the final fully connected layer was dropped . The layers used along with their respective dimensions are described in the following table:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Normalized Gray image   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x12 	|  
| Activation    		| RELU											|
| Max pooling	      	| 2x2 stride,  outputs 12x12x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x18 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 4x4x24 	|  
| Activation			| RELU											|
| Max pooling	      	| 2x2 stride,  outputs 2x2x24   				|
| Flatten	        	| Output 96    				                    |
| Fully connected		| Output 64    									|
| Activation 			|                              RELU											  									                                   |
| Fully connected	    |  Output43  |
| Softmax		  |   |


#### 3. Model training

To train the model, I used a learning rate of 0.00075 and a batch size of 256. The model was trained for 50 epochs. Type of optimizer (???)

The final training results were:
* training accuracy: 0.991
* validation accuracy: 0.971
* test accuracy: 0.947

Initially the LeNet architecture was chosen, which is an excellent "first architecture" for Convolutional Neural Networks (especially for image recognition) due to its compactness and simplicity. However the accuracy was below the project standards. In order to increase the accuracy the model was enhanced with 2 extra convolution layers (5x5) after the already existing ones. In order to avoid overfitting the last fully connected layer was dropped. All the other layers and activation functions were kept the same.

The training results provide good evidence that the model worked well. Further evidence on the effectiveness of the model can be seen in the next session, where the model is confronted with new traffic sign images found on the web.

### Model tested on new images

#### 1. German traffic signs found on the web

Here are nine German traffic signs found on the web:

![alt text][image_TestImages]

#### 2. Model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (60km/h)  | Speed limit (60km/h)							|
| Road work    			| Road work										|
| Children crossing		| Children crossing 							|
| Keep right	   		| Keep right					 				|
| Stop	        		| Stop              							|
| General caution		| General caution      		        		|  
| Bumpy road     		| Bumpy road    					    	|   
| Right-of-way at the next intersection   |  Right-of-way at the next intersection |
| No entry              | No entry                                      |


The model was able to correctly guess 9 of the 9 web traffic signs, which gives an accuracy of 100%. This is not comparable to the accuracy of the test set (0.947) but can be explained by the number of web examples used. In order to reach the accuracy of 94.7% one should test in a larger set of web images.

#### 3. Top 5 prediction probabilities

The top five soft max probabilities for each figure are:

1st image: Speed limit (60km/h)

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.00        | Speed limit (60km/h)  				     	|
| 0.000          | Speed limit (20km/h)					        |
| 0.000       	| Speed limit (30km/h)							|
| 0.000	        | Speed limit (50km/h)			 				|
| 0.000	        | Speed limit (70km/h)							|

![alt text][image_pred_1]

2nd image: Road work

| Probability [%]  |     Prediction	        					|
|:-------------:|:---------------------------------------------:|
| 100.00        | Road work				     	                |
| 0.000         | Dangerous curve to the right     				    |
| 0.000        	| Beware of ice/snow             	|
| 0.000	        | Slippery road     			 				|
| 0.000	        | Right-of-way at the next intersection		            |

![alt text][image_pred_2]

3rd image: Children crossing

| Probability [%]  |     Prediction	        					|
|:-------------:|:---------------------------------------------:|
| 100.00        | Children crossing		     	                |
| 0.000         | Road narrows on the right     	          |
| 0.000        	| Beware of ice/snow				             	|
| 0.000	        | Right-of-way at the next intersection				|
| 0.000	        | Pedestrians            |

![alt text][image_pred_3]

4th image: Keep right

| Probability [%]  |     Prediction	        					|
|:-------------:|:---------------------------------------------:|
| 100.00        | Keep right		     	                |
| 0.000         | No passing for vehicles over 3.5 metric tons    |
| 0.000       	| Priority road	             	|
| 0.000	        | Slippery road 				|
| 0.000	        | No passing           |

![alt text][image_pred_4]

5th image: Stop

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.00        | Stop 				     	|
| 0.000         | Speed limit (60km/h)	        |
| 0.000         | Speed limit (70km/h)				|
| 0.000	        | Speed limit (80km/h)		 				|
| 0.000	        | Speed limit (30km/h)			|

![alt text][image_pred_5]

6th image: General caution

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.00        | General caution 				     	|
| 0.000         | Speed limit (20km/h)				        |
| 0.000        	| Speed limit (30km/h)			|
| 0.000	        | Speed limit (50km/h)		 				|
| 0.000	        | Speed limit (60km/h)					|

![alt text][image_pred_6]

7th image: Bumpy road

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.000       | Bumpy road 				     	|
| 0.000         | Wild animals crossing			        |
| 0.000        	| Bicycles crossing				|
| 0.000	        | Speed limit (20km/h) 				|
| 0.000	        | Speed limit (30km/h)		|

![alt text][image_pred_7]

8th image: Right-of-way at the next intersection 		

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.00        | Right-of-way at the next intersection 				     	|
| 0.000         | Beware of ice/snow		        |
| 0.000        	| Speed limit (20km/h)			|
| 0.000	        | Speed limit (30km/h)				|
| 0.000	        | Speed limit (50km/h)	|

![alt text][image_pred_8]

9th image: No entry

| Probability [%]   |     Prediction	        			    |
|:-------------:|:---------------------------------------------:|
| 100.00        | No entry 				     	|
| 0.000         | Turn left ahead	        |
| 0.000        	| Turn right ahead	|
| 0.000	        | Ahead only		|
| 0.000	        | No passing|

![alt text][image_pred_9]
