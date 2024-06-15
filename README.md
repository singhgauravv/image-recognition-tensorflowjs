# Multinominal Logistic Regression - Image Recognition

## Problem Statement:

Given the pixel intensity values in an image, identify whether the character is a hand-written 0,1,2....9.

## Accuracy Achieved: 89%

## Performance Optimization in the world of JavaScript

### Memory Management

1. Memory Usage:
   (a) Create a memory snapshort to analyze the memory allocation across our program.
   (b) JavaScript `Garbage Collector` process will reclaim memory if the program reaches a state where one or more value(s) can't be referenced.
   (c) Shallow memory usage refers to the amount of memory consumed by a particular object itself, without considering the memory occupied by objects it references. It includes the memory for the object's own properties but not the objects referenced by these properties.

   ```
   user {
   name: "Alice"  -> memory usage
   age: 30        -> memory usage
   address: {     -> memory usage (only reference)
    city: "Wonderland"
    zip: "12345"
   }
   }

   ```

   (d) Retained memory usage is the total amount of memory that will be freed when a particular object is garbage collected. This includes the memory used by the object itself and the memory used by all objects that become unreachable when this object is garbage collected.

   ```
   user {
   name: "Alice"  -> memory usage
   age: 30        -> memory usage
   address: {     -> memory usage
    city: "Wonderland" -> memory usage
    zip: "12345"       -> memory usage
   }
   }

   ```

2. Minimize Memory Usage
   (a) Run ` node --inspect-brk --max-old-space-size=4096 index.js` and take a memory [snapshot](./images/memory-snapshot-pre.PNG).
   (b) Introduce loadData() fn.

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

1. **[Bayes' theorem](./images/bt.PNG)**
2. **[Marginal Probability](./images/mp.PNG)**
3. **[Conditional Probability](./images/cp.PNG)**
4. **[Softmax](./images/sm.PNG)**
5. **JavaScript Garbage Collector:** Garbage collection in JavaScript is an automatic memory management feature provided by the JavaScript engine to reclaim memory that is no longer in use, thereby preventing memory leaks and optimizing resource utilization. The primary mechanism employed is called mark-and-sweep, where the engine periodically identifies and "marks" all reachable objects starting from the root (e.g., global variables and active function calls). It then "sweeps" through memory, reclaiming space occupied by unmarked (unreachable) objects.
6. **Memory leaks:** Memory leaks in JavaScript occur when memory that is no longer needed is not released, causing the application to use increasing amounts of memory over time. This often happens due to lingering references to objects that should be garbage collected.
