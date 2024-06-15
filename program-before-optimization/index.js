require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const _ = require("lodash");
const mnist = require("mnist-data");

/** Specify the number of images to be loaded. */
const mnistData = mnist.training(0, 40000);

/** 784 features for every image */
const features = mnistData.images.values.map((image) => _.flatMap(image));

/** Encoding labels */
const encodedLabels = mnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 80,
  batchSize: 500,
});

regression.train();

const testMnistData = mnist.testing(0, 100);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("Accuracy", accuracy);
