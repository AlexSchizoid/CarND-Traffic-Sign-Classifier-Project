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

[image1]: ./examples/color_plot.png "Comparison between preprocessing"
[image2]: ./examples/distrib_test.jpg "Distrbution test samples"
[image3]: ./examples/distrib_train.jpg "Distrbution train samples"
[image4]: ./examples/distrib_valid.jpg "Distrbution valid samples"
[image5]: ./examples/learnrate.png "Learning Rate Learn curve for the Validation set"
[image6]: ./examples/processed_color.png "Processed Color Sample"
[image7]: ./examples/random_train.jpg "Random Samples Training set"
[image8]: ./examples/random_valid.jpg "Random Samples Validation set"
[image9]: ./examples/regularization.png "Compare Regularzation methods"
[image10]: ./examples/unprocessed_color.png "Unprocessed Color sample"
[image11]: ./examples/processed_gray.png "Processed Grayscale Sample"
[image12]: ./examples/new_signs.png "New signs"

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

First Visualization is a plot of various random samples from the train and validation sets.

Train set Samples:
![alt text][image7]

For the exploratory visualization part i decided to see how the various classes are distributed in the dataset. As the bar chart shows some classes are smaller than others which could possibly lead to overfitting. A solution for this might pe equalizing the number of samples per class, by fetching more examples are generating extra samples. 

The three datasets seem to have similar distributions.

![alt text][image2]
![alt text][image3]
![alt text][image4]

###Design and Test a Model Architecture

####1. I have decided to test 3 kinds of scenarios for preprocessing and in the end choose the one that seems to be working the best. 
In the first scenario, I decided to keep the samples as is, no preprocessing.
In the second scenario, I decided to keep the samples in color, but apply Contrast Limited Adaptive Histogram Equalization(CLAHE) to the Y channel of the image after converting the samples from RGB to YUV. This should create better edges for the road signs. The Last step is to convert the sample back to RGB.
In the third scenario,  the samples are grayscaled and then I apply CLAHE for the same reason as above.

The following plot shows that the second scenario seems to work a little bit better than the others, so I've chosen this method for the preprocessing step.

![alt text][image1]

Following are examples of outputs after my processing pipelines.

The original image looks like this.



![alt text][image10]

A processed color image looks like this.



![alt text][image6]

A processed grayscale image looks like this.


![alt text][image11]

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
| Dropout		| 0.5 prob|
| Fully connected		| Input = 120. Output = 84.	|
| Dropout		| 0.5 prob|
| Batch Normalization		| |
| Fully connected		| Input = 84. Output = 43.	|
| Softmax				| etc.        									|

 
 ####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 
 For the learning rate i did a comparison between different values. The plot shows that 0.003 seemed to provide the best accuracy. Using batch normalization also seems to speed up learing my allowing us to use a higher learning rate and lower number of epochs.
 
 ![alt text][image5]

To train the model i selected 100 epochs and a learning rate of 0.003 as my hyper parameters.
 
 
 ####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974  
* test set accuracy of 0.950

I started with the LeNet model since it is a simple model that has shown itself to perform reasonably well in image classification problems. The model folows the LeNet ConvNet with the addition of regularization layers of dropout and batch normalization. The regularization layers were added because the default model tended to overfit the data. The following plot shows the perfromance between the default LeNet model and the improved one with regularization. 

![alt text][image9]

To improve the model, more training data will prove useful in preventing overfitting further. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image12]

The third and fifth image might be difficult to classify since the signs are a lot smaller and placed in the right part of the picture. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection  									| 
| Priority Road 			| Priority Road 			|
| Yield					| Slippery road											|
| Stop      		| Stop Road					 				|
| Roundabout			| Go straight or right   							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

It seems the model is very sure of it's predictions, outputing 1.0 even though the 3rd and 5th predictions are clearly wrong.
I'm still investigating and wondering if this is a bug in my code when outputting the softmax probabilities.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection   									| 
| 1.0     				| Priority Road 										|
| 1.0					| Slippery road											|
| 1.0	      			| Stop Road						 				|
| 1.0				    |  Go straight or right      							|

