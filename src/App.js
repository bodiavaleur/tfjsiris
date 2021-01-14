import { useState } from "react";
import { irisModel, predict } from "./irisModel";
import * as tfvis from "@tensorflow/tfjs-vis";
import { Button, Input, Label, ParamsBlock, Title } from "./ui";

export function App() {
  const [model, setModel] = useState(null);
  const [valueToPredict, setValueToPredict] = useState([4.6, 3.4, 1.4, 0.3]);
  const [params, setParams] = useState({
    batchSize: 10,
    learningRate: 0.05,
    epochs: 20,
  });

  const trainModel = async () => {
    const trainedModel = await irisModel(params);
    setModel(trainedModel);
  };

  const predictModel = () => predict(model, valueToPredict);

  return (
    <div>
      <Title>Iris Classification</Title>
      <a href="https://archive.ics.uci.edu/ml/datasets/iris">dataset</a>
      <ParamsBlock>
        <Label>
          Batch size:
          <Input
            onChange={(e) =>
              setParams({ ...params, batchSize: +e.target.value })
            }
            defaultValue={params.batchSize}
          />
        </Label>

        <Label>
          Learning rate:
          <Input
            onChange={(e) =>
              setParams({ ...params, learningRate: +e.target.value })
            }
            defaultValue={params.learningRate}
          />
        </Label>

        <Label>
          Epochs:
          <Input
            onChange={(e) => setParams({ ...params, epochs: +e.target.value })}
            defaultValue={params.epochs}
          />
        </Label>

        <Label>
          Value to predict:
          <Input
            onChange={(e) => setValueToPredict(JSON.parse(e.target.value))}
            defaultValue={JSON.stringify(valueToPredict)}
          />
        </Label>
      </ParamsBlock>

      <Button onClick={trainModel}>Train</Button>
      <Button onClick={predictModel}>Predict</Button>
      <Button onClick={() => tfvis.visor().toggle()}>Toggle visor</Button>
    </div>
  );
}
