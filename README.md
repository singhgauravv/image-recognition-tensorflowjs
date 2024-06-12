# Multinominal Logistic Regression - Image Recognition

## Problem Statement:

Given the pixel intensity values in an image, identify whether the character is a hand-written 0,1,2....9.

## Accuracy Achieved:

## Data source: MNIST database (Modified National Institute of Standards and Technology database)

## How to encode features?

In the context of image recognition tasks using the MNIST dataset with TensorFlow.js and JavaScript, encoding features involves representing each image's pixel values in a format suitable for machine learning algorithms.

#### Encoding Pixel Values:

Each image in the MNIST dataset consists of a 28x28 grid of pixels, totaling 784 pixels per image. To encode these pixels, we flatten the 28x28 grid into a single array containing 784 elements. This array represents the grayscale values of each pixel in the image.

#### Array Organization:

The flattened pixel array for each image is then nested within an outer array. This outer array serves as a container for all the image data in the dataset.

## How to encode label values?

In our case, the total number of possible label values are going to be 10, i.e 0 to 9. To encore a optimal label values encoding,
we will create an array that will contain 1 at the index equal to the label value and otherwise 0.

For example, to represent label 5, the encoding will be [0,0,0,0,0,1,0,0,0,0].
To represent label 0, the encoding will be [1,0,0,0,0,0,0,0,0,0], and so on.

## Toolkit:

1. JavaScript
2. TensorFlowJs

## Definition

1. [Bayes' theorem](./images/bt.PNG)
2. [Marginal Probability](./images/mp.PNG)
3. [Conditional Probability](./images/cp.PNG)
4. [Softmax](./images/sm.PNG)
