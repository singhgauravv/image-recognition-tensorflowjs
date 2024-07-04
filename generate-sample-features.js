const fs = require("fs");
const mnist = require("mnist-data");
const _ = require("lodash");

const mnistData = mnist.training(0, 1);
const features = mnistData.images.values.map((image) => _.flatMap(image));

/** Prepare CSV content */
const csvContent = features
  .map((feature) => `sample_value: [${feature.join(",")}]`)
  .join("\n");

/** Write content to a CSV file */
fs.writeFile("sample-features.csv", csvContent, (err) => {
  if (err) {
    console.log("Error while writing the csv file.");
  } else {
    console.log("file write successful.");
  }
});
