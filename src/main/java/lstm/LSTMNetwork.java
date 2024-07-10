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
        initializeBiases();

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

    private void initializeBiases() {
        Arrays.fill(biasInputGate, 0.01);
        Arrays.fill(biasForgetGate, 0.01);
        Arrays.fill(biasOutputGate, 0.01);
        Arrays.fill(biasCellGate, 0.01);
        Arrays.fill(biasOutput, 0.01);
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

        if (input.length != inputSize ||
                forgetGate.length != hiddenSize ||
                inputGate.length != hiddenSize ||
                cellGate.length != hiddenSize ||
                outputGate.length != hiddenSize) {
            System.err.println("Error: Mismatched array lengths");

                System.out.println(" input length : "+input.length+" input : "+inputSize+" forgetGate length : "+forgetGate.length+" inputGate length : "+inputGate.length+
                        " cellGate length : "+cellGate.length+" outputGate length : "+outputGate.length+" hiddenSize : "+hiddenSize);

            return null;
        }

        for (int i = 0; i < cellState.length; i++) {
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

//            if (Double.isNaN(input[i])) {
//                System.err.println("Error: NaN value encountered in input at index: " + i);
//                return null;
//            }

            cellState[i] = forgetGate[i] * cellState[i] + inputGate[i] * cellGate[i];

            if (Double.isNaN(cellState[i])) {
                System.err.println("Error: NaN value encountered in cell state after update at index: " + i);
                return null;
            }
            hiddenState[i] = outputGate[i] * relu(cellState[i]);
        }

        double[] output = dotProduct(weightsOutput, hiddenState);

        if (output == null) {
            System.err.println("Error: Output is null after forward pass");
            return null;
        }

        return output;
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

//        System.out.println("output from forward : "+output);

        double[][] gradients = calculateGradients(input, hiddenState, cellState, dHiddenState, dCellState,
                dInputGate, dForgetGate, dOutputGate, dCellGate, dOutput, target, output);


        if (output == null) {
            System.err.println("Output is null");
            return;
        }

        clipGradients(gradients, 5.0);

        updateWeights(weightsOutput, gradients, learningRate);

        double[] error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            error[i] = target[i] - output[i];
        }

        double[] dOutput = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            dOutput[i] = -2 * error[i] * reluDerivative(output[i]);
        }


        error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            assert output != null;
            error[i] = target[i] - output[i];
        }

        dOutput = new double[output.length];
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

        for (int i = 0; i < hiddenSize; i++) {
            dOutputGate[i] = dHiddenState[i] * relu(cellState[i]) * reluDerivative(outputGate[i]);
            dCellState[i] += dHiddenState[i] * outputGate[i] * reluDerivative(cellState[i]);
            dInputGate[i] = dCellState[i] * cellGate[i] * reluDerivative(inputGate[i]);
            dForgetGate[i] = dCellState[i] * cellState[i] * reluDerivative(forgetGate[i]);
            dCellGate[i] = dCellState[i] * inputGate[i] * reluDerivative(cellGate[i]);
            dHiddenState[i] = dCellState[i] * forgetGate[i];
        }

        double[][] dWeightsInputGate = outerProduct(dInputGate, input);
        double[] dBiasInputGate = dInputGate.clone();

        double[][] dWeightsForgetGate = outerProduct(dForgetGate, input);
        double[] dBiasForgetGate = dForgetGate.clone();

        double[][] dWeightsOutputGate = outerProduct(dOutputGate, input);
        double[] dBiasOutputGate = dOutputGate.clone();

        double[][] dWeightsCellGate = outerProduct(dCellGate, input);
        double[] dBiasCellGate = dCellGate.clone();

        updateWeights(weightsOutput, dWeightsOutput, learningRate);
        updateBiases(biasOutput, dBiasOutput, learningRate);

        updateWeights(weightsInputGate, dWeightsInputGate, learningRate);
        updateBiases(biasInputGate, dBiasInputGate, learningRate);

        updateWeights(weightsForgetGate, dWeightsForgetGate, learningRate);
        updateBiases(biasForgetGate, dBiasForgetGate, learningRate);

        updateWeights(weightsOutputGate, dWeightsOutputGate, learningRate);
        updateBiases(biasOutputGate, dBiasOutputGate, learningRate);

        updateWeights(weightsCellGate, dWeightsCellGate, learningRate);
        updateBiases(biasCellGate, dBiasCellGate, learningRate);
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double[] relu(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = relu(x[i]);
        }
        return result;
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double[] dotProduct(double[][] matrix, double[] vector) {
        int m = matrix.length;
        int n = vector.length;
        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            result[i] = 0;
            for (int j = 0; j < n; j++) {
                result[i] += matrix[i][j] * vector[j];
//                System.out.println(i+" "+j+" result : "+result[i]+" = "+matrix[i][j]+" * "+vector[j]);
            }
        }

        int i = 1;

        for (double row : result) {

//            System.out.println(i+" : "+Arrays.toString(new double[]{row}));
            i++;

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

    private double[][] calculateGradients(double[] input, double[] hiddenState, double[] cellState, double[] dHiddenState, double[] dCellState,
                                          double[] dInputGate, double[] dForgetGate, double[] dOutputGate, double[] dCellGate, double[] dOutput,
                                          double[] target, double[] output) {
        double[][] gradients = new double[9][];

        gradients[0] = dHiddenState;
        gradients[1] = dCellState;
        gradients[2] = dInputGate;
        gradients[3] = dForgetGate;
        gradients[4] = dOutputGate;
        gradients[5] = dCellGate;
        gradients[6] = dOutput;

        gradients[7] = new double[hiddenSize * inputSize];
        gradients[8] = new double[hiddenSize * hiddenSize];

        double[] extendedDOutput = new double[hiddenSize];
        Arrays.fill(extendedDOutput, dOutput[0]);

        for (int i = 0; i < target.length; i++) {
            extendedDOutput[i] = -2 * (target[i] - output[i]) * reluDerivative(output[i]);
        }

        for (int i = 0; i < dHiddenState.length; i++) {
            dHiddenState[i] = extendedDOutput[i] * leakyReluDerivative(hiddenState[i]);
            dCellState[i] = dHiddenState[i] * dOutputGate[i] * leakyReluDerivative(cellState[i]);
            dInputGate[i] = dCellState[i] * dCellGate[i] * leakyReluDerivative(dInputGate[i]);
            dForgetGate[i] = dCellState[i] * cellState[i] * leakyReluDerivative(dForgetGate[i]);
            dOutputGate[i] = dHiddenState[i] * leakyReluDerivative(dOutputGate[i]);
            dCellGate[i] = dCellState[i] * dInputGate[i] * leakyReluDerivative(dCellGate[i]);
        }

        for (int i = 0; i < gradients[7].length; i++) {
            gradients[7][i] = 0;
        }

        for (int i = 0; i < gradients[8].length; i++) {
            gradients[8][i] = 0;
        }

        return gradients;
    }


    private void clipGradients(double[][] gradients, double threshold) {
        for (double[] gradient : gradients) {
            for (int i = 0; i < gradient.length; i++) {
                if (gradient[i] > threshold) {
                    gradient[i] = threshold;
                } else if (gradient[i] < -threshold) {
                    gradient[i] = -threshold;
                }
            }
        }
    }



    private void updateWeights(double[][] weights, double[][] gradients, double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * gradients[i][j];
            }
        }
    }

    private void updateBiases(double[] biases, double[] gradients, double learningRate) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] -= learningRate * gradients[i];
        }
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
