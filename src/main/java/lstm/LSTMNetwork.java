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

        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];

        // Initialize weights with small random values
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weightsInputGate[i][j] = random.nextGaussian() * 0.01;
                weightsForgetGate[i][j] = random.nextGaussian() * 0.01;
                weightsOutputGate[i][j] = random.nextGaussian() * 0.01;
                weightsCellGate[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsHiddenInputGate[i][j] = random.nextGaussian() * 0.01;
                weightsHiddenForgetGate[i][j] = random.nextGaussian() * 0.01;
                weightsHiddenOutputGate[i][j] = random.nextGaussian() * 0.01;
                weightsHiddenCellGate[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            biasInputGate[i] = random.nextGaussian() * 0.01;
            biasForgetGate[i] = random.nextGaussian() * 0.01;
            biasOutputGate[i] = random.nextGaussian() * 0.01;
            biasCellGate[i] = random.nextGaussian() * 0.01;
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsOutput[i][j] = random.nextGaussian() * 0.01;
            }
            biasOutput[i] = random.nextGaussian() * 0.01;
        }
    }

    private void initializeBiases(double[] biases) {
        Arrays.fill(biases, 0.1);
    }

    public double[] forward(double[] input, double[] hiddenState, double[] cellState) {
        inputGate = relu(add(dotProduct(weightsInputGate, input), dotProduct(weightsHiddenInputGate, hiddenState), biasInputGate));
        forgetGate = relu(add(dotProduct(weightsForgetGate, input), dotProduct(weightsHiddenForgetGate, hiddenState), biasForgetGate));
        outputGate = relu(add(dotProduct(weightsOutputGate, input), dotProduct(weightsHiddenOutputGate, hiddenState), biasOutputGate));
        cellGate = relu(add(dotProduct(weightsCellGate, input), dotProduct(weightsHiddenCellGate, hiddenState), biasCellGate));

        if (input.length != inputSize ||
                forgetGate.length != hiddenSize ||
                inputGate.length != hiddenSize ||
                cellGate.length != hiddenSize ||
                outputGate.length != hiddenSize) {
            System.err.println("Error: Mismatched array lengths");
            return null;
        }

        for (int i = 0; i < cellState.length; i++) {

//            System.out.println(i + " inputs : " + input[0] + " , " + input[1] + " , " + input[2] + " , " + input[3] + " , " + input[4] + " , " + input[5] + " , " + input[6] + " , " + input[7]);

            if (Double.isNaN(forgetGate[i])) {
                System.err.println("Error: NaN value encountered in forget gate at index: " + i);
                return null;
            }

            if (Double.isNaN(inputGate[i])) {
                System.err.println("Error: NaN value encountered in input gate at index: " + i);
                return null;
            }

            if (Double.isNaN(cellGate[i])) {
                System.err.println("Error: NaN value encountered in cell gate at index: " + i);
                return null;
            }

            if (Double.isNaN(input[1])) {
                System.err.println("Error: NaN value encountered in input at index: " + i);
                return null;

            }

            cellState[i] = forgetGate[i] * cellState[i] + inputGate[i] * cellGate[i];

            if (Double.isNaN(cellState[i])) {
                System.err.println("Error: NaN value encountered in cell state after update at index: " + i);
                return null;
            }
            hiddenState[i] = outputGate[i] * relu(cellState[i]);
        }

        double[] output = dotProduct(weightsOutput, hiddenState);

        // Adding null check for output
        if (output == null) {
            System.err.println("Error: Output is null after forward pass");
            return null;
        }

        return output;
    }

    private double[] add(double[] a, double[] b, double[] c, double[] d) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i] + c[i] + d[i];
        }
        return result;
    }

    public void backpropagate(double[] input, double[] target, double learningRate) {
        double[] hiddenState = new double[hiddenSize];
        double[] cellState = new double[hiddenSize];
        double[] output = forward(input, hiddenState, cellState);  // Ensure this returns a non-null value

        if (output == null) {
            System.err.println("Output is null");
        }

        double[] error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            assert output != null;
            error[i] = target[i] - output[i];
        }

        double[] dOutput = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            dOutput[i] = -2 * error[i] * reluDerivative(output[i]);
        }

        double[][] dWeightsOutput = outerProduct(dOutput, hiddenState);
        double[] dBiasOutput = dOutput.clone();

        double[] dHiddenState = dotProductTranspose(weightsOutput, dOutput);

        double[] dCellState = new double[hiddenSize];
        double[] dInputGate = new double[hiddenSize];
        double[] dForgetGate = new double[hiddenSize];
        double[] dOutputGate = new double[hiddenSize];
        double[] dCellGate = new double[hiddenSize];

        for (int t = hiddenSize - 1; t >= 0; t--) {
            double[] dOutputGateTemp = new double[hiddenSize];
            double[] dCellStateTemp = new double[hiddenSize];
            double[] dInputGateTemp = new double[hiddenSize];
            double[] dForgetGateTemp = new double[hiddenSize];
            double[] dCellGateTemp = new double[hiddenSize];

            for (int i = 0; i < hiddenSize; i++) {
                dOutputGateTemp[i] = dHiddenState[i] * relu(cellState[i]) * reluDerivative(outputGate[i]);
                dCellStateTemp[i] = dHiddenState[i] * outputGate[i] * reluDerivative(cellState[i]) + dCellState[i];
                dInputGateTemp[i] = dCellStateTemp[i] * cellGate[i] * reluDerivative(inputGate[i]);
                dForgetGateTemp[i] = dCellStateTemp[i] * cellState[i] * reluDerivative(forgetGate[i]);
                dCellGateTemp[i] = dCellStateTemp[i] * inputGate[i] * reluDerivative(cellGate[i]);

                dCellState[i] += dCellStateTemp[i];
            }

            dInputGate = add(dInputGate, dInputGateTemp);
            dForgetGate = add(dForgetGate, dForgetGateTemp);
            dOutputGate = add(dOutputGate, dOutputGateTemp);
            dCellGate = add(dCellGate, dCellGateTemp);

            dHiddenState = add(dotProductTranspose(weightsHiddenInputGate, dInputGateTemp),
                    dotProductTranspose(weightsHiddenForgetGate, dForgetGateTemp),
                    dotProductTranspose(weightsHiddenOutputGate, dOutputGateTemp),
                    dotProductTranspose(weightsHiddenCellGate, dCellGateTemp));
        }

        double[][] dWeightsInputGate = outerProduct(dInputGate, input);
        double[][] dWeightsForgetGate = outerProduct(dForgetGate, input);
        double[][] dWeightsOutputGate = outerProduct(dOutputGate, input);
        double[][] dWeightsCellGate = outerProduct(dCellGate, input);
        double[][] dWeightsHiddenInputGate = outerProduct(dInputGate, hiddenState);
        double[][] dWeightsHiddenForgetGate = outerProduct(dForgetGate, hiddenState);
        double[][] dWeightsHiddenOutputGate = outerProduct(dOutputGate, hiddenState);
        double[][] dWeightsHiddenCellGate = outerProduct(dCellGate, hiddenState);

        updateWeights(weightsInputGate, dWeightsInputGate, learningRate);
        updateWeights(weightsForgetGate, dWeightsForgetGate, learningRate);
        updateWeights(weightsOutputGate, dWeightsOutputGate, learningRate);
        updateWeights(weightsCellGate, dWeightsCellGate, learningRate);
        updateWeights(weightsHiddenInputGate, dWeightsHiddenInputGate, learningRate);
        updateWeights(weightsHiddenForgetGate, dWeightsHiddenForgetGate, learningRate);
        updateWeights(weightsHiddenOutputGate, dWeightsHiddenOutputGate, learningRate);
        updateWeights(weightsHiddenCellGate, dWeightsHiddenCellGate, learningRate);
        updateWeights(weightsOutput, dWeightsOutput, learningRate);

        updateBiases(biasInputGate, dInputGate, learningRate);
        updateBiases(biasForgetGate, dForgetGate, learningRate);
        updateBiases(biasOutputGate, dOutputGate, learningRate);
        updateBiases(biasCellGate, dCellGate, learningRate);
        updateBiases(biasOutput, dBiasOutput, learningRate);
    }


    private double[] relu(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.max(0, x[i]);
        }
        return x;
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double[] tanh(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.tanh(x[i]);
        }
        return x;
    }

    private double tanhDerivative(double x) {
        return 1 - Math.tanh(x) * Math.tanh(x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double[] dotProduct(double[][] weights, double[] inputs) {
        double[] result = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < inputs.length; j++) {
                result[i] += weights[i][j] * inputs[j];
            }
        }
        return result;
    }

    private double[] dotProductTranspose(double[][] weights, double[] dOutput) {
        double[] result = new double[weights[0].length];
        for (int i = 0; i < weights[0].length; i++) {
            for (int j = 0; j < dOutput.length; j++) {
                result[i] += dOutput[j] * weights[j][i];
            }
        }
        return result;
    }

    private double[][] outerProduct(double[] vec1, double[] vec2) {
        double[][] result = new double[vec1.length][vec2.length];
        for (int i = 0; i < vec1.length; i++) {
            for (int j = 0; j < vec2.length; j++) {
                result[i][j] = vec1[i] * vec2[j];
            }
        }
        return result;
    }

    private double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private double[] add(double[] a, double[] b, double[] c) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i] + c[i];
        }
        return result;
    }

    private void updateWeights(double[][] weights, double[][] dWeights, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * dWeights[i][j];
            }
        }
    }

    private void updateBiases(double[] biases, double[] dBiases, double learningRate) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * dBiases[i];
        }
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
