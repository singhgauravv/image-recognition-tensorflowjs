require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const _ = require("lodash");
const mnist = require("mnist-data");
const observations = require("./img-for-prediction");

/** Optimization: Release the reference to actual mnistData (mnist.training(0, 1000)) once it has been
 *  used to make it eligible for garbage collection. Introduce loadData() fn, return features and labels,
 *  and once exited, there will be no reference to the data mnist.training(0, 1000)
 *  (what mnistData is referring to).
 */

function loadData() {
  /**Specify the number of images to be loaded.*/
  const mnistData = mnist.training(0, 40000);

  /** 784 features for every image */
  const features = mnistData.images.values.map((image) => _.flatMap(image));

  /** Encoding labels */
  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

/** Create a new instance of LogisticRegression class. */
const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 500,
});

/** Train the model. */
regression.train();

/** Prepare test features & labels, and calculate the accuracy of the model. */
const testMnistData = mnist.testing(0, 100);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

/** Print the accuracy. */
const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("Accuracy", accuracy);

/** Make a prediction using the trained model. Change img-for-prediction.js to predict using another image.*/
const prediction = regression.predict(observations);
console.log("The image that you uploaded is", prediction.arraySync());
