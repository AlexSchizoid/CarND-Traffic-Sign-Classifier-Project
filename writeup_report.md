#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AlexSchizoid/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Pyhton's functionality to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

For the exploratory visualization part i decided to see how the various classes are distributed in the dataset. As the bar chart shows some classes are smaller than others which could possibly lead to overfitting. A solution for this might pe equalizing the number of samples per class, by fetching more examples are generating extra samples. 

The three datasets seem to have similar distributions.

![alt text][image1]

###Design and Test a Model Architecture

####1. I have decided to test 3 kinds of scenarios for preprocessing and in the end choose the one that seems to be working the best. 
In the first scenario, I decided to keep the samples as is, no preprocessing.
In the second scenario, I decided to keep the samples in color, but apply Contrast Limited Adaptive Histogram Equalization(CLAHE) to the Y channel of the image after converting the samples from RGB to YUV. This should create better edges for the road signs. The Last step is to convert the sample back to RGB.
In the third scenario,  the samples are grayscaled and then I apply CLAHE for the same reason as above.

The following plot shows that the second scenario seems to work a little bit better than the others, so I've chosen this method for the preprocessing step.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| input depth 1/3(depending if color or gray) output depth 6, strides=[1, 1, 1, 1]. Output = 28x28x6.
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| input depth 6 output depth 6. strides=[1, 1, 1, 1], Output = 10x10x16.
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input = 400. Output = 120.	|
| Dropout		| |
| Fully connected		| Input = 120. Output = 84.	|
| Dropout		| |
| Batch Normalization		| |
| Fully connected		| Input = 84. Output = 43.	|
| Softmax				| etc.        									|

 
 ####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 
 For the learning rate i did a comparison between different values. The plot shows that 0.003 seemed to provide the best accuracy. Using batch normalization also seems to speed up learing my allowing us to use a higher learning rate and lower number of epochs.

To train the model i selected 100 epochs and a learning rate of 0.003 as my hyper parameters.
 
 
 ####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974  
* test set accuracy of 0.950

I started with the LeNet model since it is a simple model that has shown itself to perform reasonably well in image classification problems. The model folows the LeNet ConvNet with the addition of regularization layers of dropout and batch normalization. The regularization layers were added because the default model tended to overfit the data. The following plot shows the perfromance between the default LeNet model and the improved one with regularization. 

To improve the model, more training data will prove useful in preventing overfitting further. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third and fifth image might be difficult to classify since the signs are a lot smaller and placed in the right part of the picture. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection  									| 
| Priority Road 			| Priority Road 			|
| Yield					| Slippery road											|
| Stop      		| Stop Road					 				|
| Roundabout			| No passing for vehicles over 3.5 metric tons    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Right-of-way at the next intersection   									| 
| .20     				| Priority Road 										|
| .05					| Slippery road											|
| .04	      			| Stop Road						 				|
| .01				    | No passing for vehicles over 3.5 metric tons       							|

