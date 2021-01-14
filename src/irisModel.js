import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

export async function irisModel(modelParams) {
  const { batchSize, learningRate, epochs } = modelParams;

  /* Load dataset */
  const csvUrl = "datasets/iris.csv";
  const csvDataset = tf.data.csv(csvUrl, {
    columnConfigs: {
      species: {
        isLabel: true,
      },
    },
  });

  /* Convert loaded dataset */

  // Get number of column names and subtract 1 for label of the column
  const countOfFeatures = (await csvDataset.columnNames()).length - 1;
  const convertedData = csvDataset
    .map(({ xs, ys }) => {
      // Use one-hot encoding
      const labels = [
        ys.species === "setosa" ? 1 : 0,
        ys.species === "virginica" ? 1 : 0,
        ys.species === "versicolor" ? 1 : 0,
      ];
      return { xs: Object.values(xs), ys: Object.values(labels) };
    })
    .batch(batchSize);

  /* Define the model */
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [countOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );
  model.add(tf.layers.dense({ activation: "softmax", units: 3 }));
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
  });

  /* Fit dataset */

  // For visualizing the loss
  const surface = { name: "Loss", tab: "Training" };
  const history = [];

  await model.fitDataset(convertedData, {
    epochs: epochs,
    callbacks: {
      onTrainEnd: () => alert("Training is done. Now you can predict a value"),
      onEpochEnd: (e, logs) => {
        history.push(logs);
        tfvis.show.history(surface, history, ["loss"]);
        console.log(`Epoch: ${e} | Loss: ${logs.loss}`);
      },
    },
  });

  return model;
}

export function predict(model, value) {
  if (value.length !== 4) {
    alert("Tensor should have 4 values");
    return;
  }

  if (model) {
    const labels = ["Setosa", "Virginica", "Versicolor"];
    const predictValues = tf.tensor2d(value, [1, 4]);
    const modelPrediction = tf
      .argMax(model.predict(predictValues), 1)
      .dataSync();

    alert(labels[modelPrediction]);
  } else {
    alert("First you need to train a model");
  }
}
