package lstm;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class LSTMNetwork implements Serializable {
    private double[] inputGate;
    private double[] forgetGate;
    private double[] outputGate;
    private double[] cellGate;
    private double[] hiddenState;
    private double[] cellState;

    private double[][] weightsInputGate;
    private double[][] weightsForgetGate;
    private double[][] weightsOutputGate;
    private double[][] weightsCellGate;

    private double[][] weightsHiddenInputGate;
    private double[][] weightsHiddenForgetGate;
    private double[][] weightsHiddenOutputGate;
    private double[][] weightsHiddenCellGate;

    private double[] biasInputGate;
    private double[] biasForgetGate;
    private double[] biasOutputGate;
    private double[] biasCellGate;

    private double[][] weightsOutput;
    private double[] biasOutput;

    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    private double[] dHiddenState;
    private double[] dCellState;
    private double[] dInputGate;
    private double[] dForgetGate;
    private double[] dOutputGate;
    private double[] dCellGate;
    private double[] dOutput;

    public LSTMNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // Initialize weights and biases
        initializeWeights();

        // Initialize biases with specific values
        initializeBiases(biasInputGate);
        initializeBiases(biasForgetGate);
        initializeBiases(biasOutputGate);
        initializeBiases(biasCellGate);
        initializeBiases(biasOutput);

        // Initialize hidden and cell states
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];
    }

    private void initializeWeights() {
        Random random = new Random();

        weightsInputGate = new double[hiddenSize][inputSize];
        weightsForgetGate = new double[hiddenSize][inputSize];
        weightsOutputGate = new double[hiddenSize][inputSize];
        weightsCellGate = new double[hiddenSize][inputSize];

        weightsHiddenInputGate = new double[hiddenSize][hiddenSize];
        weightsHiddenForgetGate = new double[hiddenSize][hiddenSize];
        weightsHiddenOutputGate = new double[hiddenSize][hiddenSize];
        weightsHiddenCellGate = new double[hiddenSize][hiddenSize];

        biasInputGate = new double[hiddenSize];
        biasForgetGate = new double[hiddenSize];
        biasOutputGate = new double[hiddenSize];
        biasCellGate = new double[hiddenSize];

        weightsOutput = new double[outputSize][hiddenSize];
        biasOutput = new double[outputSize];

        // Initialize weights with small random values
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize));
                weightsForgetGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize));
                weightsOutputGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize));
                weightsCellGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize));
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenInputGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + hiddenSize));
                weightsHiddenForgetGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + hiddenSize));
                weightsHiddenOutputGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + hiddenSize));
                weightsHiddenCellGate[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + hiddenSize));
            }
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsOutput[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + outputSize));
            }
            biasOutput[i] = random.nextGaussian() * Math.sqrt(2.0 / (hiddenSize + outputSize));
        }
    }

    private void initializeBiases(double[] biases) {
        Arrays.fill(biases, 0.01);
    }

    private double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    private double[] leakyRelu(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = leakyRelu(x[i]);
        }
        return result;
    }

    private double leakyReluDerivative(double x) {
        return x > 0 ? 1 : 0.01;
    }

    private double[] leakyReluDerivative(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = leakyReluDerivative(x[i]);
        }
        return result;
    }

    public double[] forward(double[] input, double[] hiddenState, double[] cellState) {
        inputGate = leakyRelu(add(dotProduct(weightsInputGate, input), dotProduct(weightsHiddenInputGate, hiddenState), biasInputGate));
        forgetGate = leakyRelu(add(dotProduct(weightsForgetGate, input), dotProduct(weightsHiddenForgetGate, hiddenState), biasForgetGate));
        outputGate = leakyRelu(add(dotProduct(weightsOutputGate, input), dotProduct(weightsHiddenOutputGate, hiddenState), biasOutputGate));
        cellGate = leakyRelu(add(dotProduct(weightsCellGate, input), dotProduct(weightsHiddenCellGate, hiddenState), biasCellGate));

        for (int i = 0; i < cellState.length; i++) {
            cellState[i] = forgetGate[i] * cellState[i] + inputGate[i] * cellGate[i];
            hiddenState[i] = outputGate[i] * leakyRelu(cellState[i]);
        }

        return dotProduct(weightsOutput, hiddenState);
    }

    private double[] add(double[] a, double[] b, double[] c) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i] + c[i];
        }
        return result;
    }

    private double[] add(double[] a, double[] b, double[] c, double[] d) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i] + c[i] + d[i];
        }
        return result;
    }

    public void backpropagate(double[] input, double[] target, double learningRate) {
        dHiddenState = new double[hiddenSize];
        dCellState = new double[hiddenSize];
        dInputGate = new double[hiddenSize];
        dForgetGate = new double[hiddenSize];
        dOutputGate = new double[hiddenSize];
        dCellGate = new double[hiddenSize];
        dOutput = new double[outputSize];

        double[] hiddenState = new double[hiddenSize];
        double[] cellState = new double[hiddenSize];
        double[] output = forward(input, hiddenState, cellState);

        double[] error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            error[i] = target[i] - output[i];
        }

        for (int i = 0; i < output.length; i++) {
            dOutput[i] = -2 * error[i] * leakyReluDerivative(output[i]);
        }

        double[][] dWeightsOutput = outerProduct(dOutput, hiddenState);
        double[] dBiasOutput = dOutput.clone();
        double[] dHiddenState = dotProductTranspose(weightsOutput, dOutput);

        double[] dCellState = new double[hiddenSize];
        double[] dInputGate = new double[hiddenSize];
        double[] dForgetGate = new double[hiddenSize];
        double[] dOutputGate = new double[hiddenSize];
        double[] dCellGate = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            dOutputGate[i] = dHiddenState[i] * leakyRelu(cellState[i]) * leakyReluDerivative(outputGate[i]);
            dCellState[i] += dHiddenState[i] * outputGate[i] * leakyReluDerivative(cellState[i]);
            dInputGate[i] = dCellState[i] * cellGate[i] * leakyReluDerivative(inputGate[i]);
            dForgetGate[i] = dCellState[i] * cellState[i] * leakyReluDerivative(forgetGate[i]);
            dCellGate[i] = dCellState[i] * inputGate[i] * leakyReluDerivative(cellGate[i]);
        }

        double[][] dWeightsInputGate = outerProduct(dInputGate, input);
        double[][] dWeightsForgetGate = outerProduct(dForgetGate, input);
        double[][] dWeightsOutputGate = outerProduct(dOutputGate, input);
        double[][] dWeightsCellGate = outerProduct(dCellGate, input);

        double[][] dWeightsHiddenInputGate = outerProduct(dInputGate, hiddenState);
        double[][] dWeightsHiddenForgetGate = outerProduct(dForgetGate, hiddenState);
        double[][] dWeightsHiddenOutputGate = outerProduct(dOutputGate, hiddenState);
        double[][] dWeightsHiddenCellGate = outerProduct(dCellGate, hiddenState);

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputGate[i][j] -= learningRate * dWeightsInputGate[i][j];
                weightsForgetGate[i][j] -= learningRate * dWeightsForgetGate[i][j];
                weightsOutputGate[i][j] -= learningRate * dWeightsOutputGate[i][j];
                weightsCellGate[i][j] -= learningRate * dWeightsCellGate[i][j];
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenInputGate[i][j] -= learningRate * dWeightsHiddenInputGate[i][j];
                weightsHiddenForgetGate[i][j] -= learningRate * dWeightsHiddenForgetGate[i][j];
                weightsHiddenOutputGate[i][j] -= learningRate * dWeightsHiddenOutputGate[i][j];
                weightsHiddenCellGate[i][j] -= learningRate * dWeightsHiddenCellGate[i][j];
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            biasInputGate[i] -= learningRate * dInputGate[i];
            biasForgetGate[i] -= learningRate * dForgetGate[i];
            biasOutputGate[i] -= learningRate * dOutputGate[i];
            biasCellGate[i] -= learningRate * dCellGate[i];
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsOutput[i][j] -= learningRate * dWeightsOutput[i][j];
            }
            biasOutput[i] -= learningRate * dBiasOutput[i];
        }
    }

    private double[] dotProduct(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private double[] dotProductTranspose(double[][] matrix, double[] vector) {
        double[] result = new double[matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                result[j] += matrix[i][j] * vector[i];
            }
        }
        return result;
    }

    private double[][] outerProduct(double[] vectorA, double[] vectorB) {
        double[][] result = new double[vectorA.length][vectorB.length];
        for (int i = 0; i < vectorA.length; i++) {
            for (int j = 0; j < vectorB.length; j++) {
                result[i][j] = vectorA[i] * vectorB[j];
            }
        }
        return result;
    }


    public void resetState() {
        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];
    }

    public void saveModel(String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static LSTMNetwork loadModel(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            System.out.println("Model loading");
            return (LSTMNetwork) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Creating model");
            return null;
        }
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public double[] getHiddenState() {
        return hiddenState;
    }

    public double[] getCellState() {
        return cellState;
    }
}
