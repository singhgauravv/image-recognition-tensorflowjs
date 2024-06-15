# Multinominal Logistic Regression - Image Recognition

## How to Run:

Run `node index.js` in any of the two directories to check the accuracy of the model.

## Problem Statement:

Given the pixel intensity values in an image, identify whether the character is a hand-written 0,1,2....9.

## Accuracy Achieved: 96%

## Memory Usage Improvement: 90% (before optimization 14 GB, after 1.6 GB)

## Memory Snapshorts: [Before](./images/before-opti.PNG) & [After](./images//after-opti.PNG) Optimization

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
   (b) Introduce loadData() fn to optimize memory usage during data loading.
   (c) Tensorflow Memory Usage: It holds reference to every tensor that gets created during the program run.
   (d) Use tf.tidy() cleans up tensors automatically inside it. If tensors need to be maintianed, they must be returned
   from this function.
   (e)

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
7. `tf.ENV.registry.webgl.backend.texData.data` stores metadata and references to all GPU textures created during a TensorFlow.js WebGL backend session. Proper tensor disposal ensures efficient GPU memory usage and prevents memory leaks.
8. **WeakMap :** A WeakMap in JavaScript is a collection of key-value pairs where the keys are objects and the values can be arbitrary values. The key feature of a WeakMap is that it allows for garbage collection of the keys. If there are no other references to an object used as a key in a WeakMap, the key-value pair can be garbage collected, which helps in managing memory efficiently.

   **Key Characteristics of WeakMap**

(a) Garbage Collection: If an object used as a key in a WeakMap has no other references, it can be garbage collected.

(b) Keys Must Be Objects: Unlike regular maps, keys in a WeakMap must be objects, not primitive values.

(c) Non-Enumerable: WeakMaps do not expose their keys and do not provide a way to iterate over their entries.

(d) No Clear Method: WeakMaps do not have a clear method to remove all entries.

9. **[Cross Entropy:](./images/ce.PNG)** Cross entropy is a loss function used in classification problems, particularly for measuring the performance of a model whose output is a probability value between 0 and 1. It is commonly used in binary classification (as binary cross entropy) and multi-class classification (as categorical cross entropy).
