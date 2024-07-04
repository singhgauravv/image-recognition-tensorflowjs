/** Standard imports. */
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

/** Main Logistic Regression class that exposes different methods in the context of the model.
 * @param features: 28*28 pixels images.
 * @param labels: labels.
 * @param options: model attributes.
 */
class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    /** cost function, scientific referring to cross entropy */
    this.costHistory = [];

    this.options = Object.assign(
      /**default model attributes */
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
        decisionBoundary: 0.5,
      },
      options
    );

    /** Initialize weights. */
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    /**
     * Calculate the gradients of the Mean Squared Error (MSE) with respect to weights (including bias).
     * Formula: (transpose of Features * ((Features * Weights) - Labels)) / n * 2
     */
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose() // Transpose of the feature matrix
      .matMul(differences) // Dot product with prediction errors
      .div(features.shape[0]); // Average over the number of data points

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }
  /** Method to train the model. */
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      /** Implement batch gradient descent using tensorflowjs. */
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );

          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      debugger;
      this.recordCost();
      this.updateLearningRate();
    }
  }

  /** Method to make a prediction using the trained model.
   * @param: A 2D array representing image pixels.
   */
  predict(observations) {
    /** Standardize observations */
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions.notEqual(testLabels).sum().arraySync();

    /** % correct:  (total predictions - Incorrect Predictions)/ total predictions */
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    /** Backfilling variance */
    const filler = variance.cast("bool").logicalNot().cast("float32");

    this.mean = mean;
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {
      /** Calculate cross entropy. */
      const guesses = this.features.matMul(this.weights).sigmoid();
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());

      const termTwo = this.labels.mul(-1).add(1).transpose().matMul(
        guesses
          .mul(-1)
          .add(1)
          .add(1e-7) // Add a constant to avoid log(0).
          .log()
      );

      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync();
    });

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
