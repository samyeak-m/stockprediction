package lstm;

import java.util.Random;

public class LSTMNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;

    private double[] hiddenState;
    private double[] cellState;

    private double[][] Wf, Wi, Wo, Wc, Wy;
    private double[] bf, bi, bo, bc, by;

    // Store the gate activations and states for the backward pass
    private double[] inputGate, forgetGate, outputGate, cellGate;
    private double[] prevHiddenState, prevCellState;

    public LSTMNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.hiddenState = new double[hiddenSize];
        this.cellState = new double[hiddenSize];

        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();

        Wf = new double[hiddenSize][inputSize + hiddenSize];
        Wi = new double[hiddenSize][inputSize + hiddenSize];
        Wo = new double[hiddenSize][inputSize + hiddenSize];
        Wc = new double[hiddenSize][inputSize + hiddenSize];
        Wy = new double[outputSize][hiddenSize];

        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bo = new double[hiddenSize];
        bc = new double[hiddenSize];
        by = new double[outputSize];

        // Initialize weights with small random values
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                Wf[i][j] = rand.nextGaussian() * 0.1;
                Wi[i][j] = rand.nextGaussian() * 0.1;
                Wo[i][j] = rand.nextGaussian() * 0.1;
                Wc[i][j] = rand.nextGaussian() * 0.1;
            }
            bf[i] = rand.nextGaussian() * 0.1;
            bi[i] = rand.nextGaussian() * 0.1;
            bo[i] = rand.nextGaussian() * 0.1;
            bc[i] = rand.nextGaussian() * 0.1;
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                Wy[i][j] = rand.nextGaussian() * 0.1;
            }
            by[i] = rand.nextGaussian() * 0.1;
        }
    }

    public double[] forward(double[] input, double[] prevHiddenState, double[] prevCellState) {
        this.prevHiddenState = prevHiddenState.clone();
        this.prevCellState = prevCellState.clone();

        double[] combined = new double[input.length + prevHiddenState.length];
        System.arraycopy(input, 0, combined, 0, input.length);
        System.arraycopy(prevHiddenState, 0, combined, input.length, prevHiddenState.length);

        forgetGate = sigmoid(add(matrixVectorMultiply(Wf, combined), bf));
        inputGate = sigmoid(add(matrixVectorMultiply(Wi, combined), bi));
        outputGate = sigmoid(add(matrixVectorMultiply(Wo, combined), bo));
        cellGate = tanh(add(matrixVectorMultiply(Wc, combined), bc));

        cellState = add(elementwiseMultiply(forgetGate, prevCellState), elementwiseMultiply(inputGate, cellGate));
        hiddenState = elementwiseMultiply(outputGate, tanh(cellState));

        double[] output = add(matrixVectorMultiply(Wy, hiddenState), by);

        return output;
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return result;
    }

    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
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

    private double[] elementwiseMultiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
}
