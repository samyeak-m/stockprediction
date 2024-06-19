package lstm;

import java.io.*;
import java.util.Random;

public class LSTMNetwork implements Serializable {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final double[][] Wf;
    private final double[][] Wi;
    private final double[][] Wo;
    private final double[][] Wc;
    private final double[][] Wy;
    private final double[] bf;
    private final double[] bi;
    private final double[] bo;
    private final double[] bc;
    private final double[] by;

    private double[] hiddenState;
    private double[] cellState;

    private double[] ft;
    private double[] it;
    private double[] ot;
    private double[] ct_hat;

    private final double clipThreshold = 5.0;

    public LSTMNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

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

        hiddenState = new double[hiddenSize];
        cellState = new double[hiddenSize];

        ft = new double[hiddenSize];
        it = new double[hiddenSize];
        ot = new double[hiddenSize];
        ct_hat = new double[hiddenSize];

        Random rand = new Random();
        initializeWeights(rand);
    }

    private void initializeWeights(Random rand) {
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

    public double[] forward(double[] input, double[] hiddenState, double[] cellState) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must match the network input size");
        }
        if (hiddenState.length != hiddenSize || cellState.length != hiddenSize) {
            throw new IllegalArgumentException("Hidden state and cell state sizes must match the network hidden size");
        }

        double[] combined = new double[input.length + hiddenState.length];
        System.arraycopy(input, 0, combined, 0, input.length);
        System.arraycopy(hiddenState, 0, combined, input.length, hiddenState.length);

        ft = sigmoid(add(matVecMul(Wf, combined), bf));
        it = sigmoid(add(matVecMul(Wi, combined), bi));
        ot = sigmoid(add(matVecMul(Wo, combined), bo));
        ct_hat = tanh(add(matVecMul(Wc, combined), bc));

        double[] newCellState = new double[cellState.length];
        for (int i = 0; i < cellState.length; i++) {
            newCellState[i] = ft[i] * cellState[i] + it[i] * ct_hat[i];
        }

        double[] newHiddenState = new double[hiddenState.length];
        for (int i = 0; i < hiddenState.length; i++) {
            newHiddenState[i] = ot[i] * Math.tanh(newCellState[i]);
        }

        double[] output = add(matVecMul(Wy, newHiddenState), by);

        System.arraycopy(newCellState, 0, cellState, 0, cellState.length);
        System.arraycopy(newHiddenState, 0, hiddenState, 0, hiddenState.length);

        return output;
    }

    public void clipGradients(double threshold) {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize + hiddenSize; j++) {
                Wf[i][j] = Math.min(Wf[i][j], threshold);
                Wi[i][j] = Math.min(Wi[i][j], threshold);
                Wo[i][j] = Math.min(Wo[i][j], threshold);
                Wc[i][j] = Math.min(Wc[i][j], threshold);
            }
            bf[i] = Math.min(bf[i], threshold);
            bi[i] = Math.min(bi[i], threshold);
            bo[i] = Math.min(bo[i], threshold);
            bc[i] = Math.min(bc[i], threshold);
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                Wy[i][j] = Math.min(Wy[i][j], threshold);
            }
            by[i] = Math.min(by[i], threshold);
        }
    }

    public void backpropagate(double[] input, double[] target, double learningRate) {
        double[] combined = new double[input.length + hiddenState.length];
        System.arraycopy(input, 0, combined, 0, input.length);
        System.arraycopy(hiddenState, 0, combined, input.length, hiddenState.length);

        double[] ft = sigmoid(add(matVecMul(Wf, combined), bf));
        double[] it = sigmoid(add(matVecMul(Wi, combined), bi));
        double[] ot = sigmoid(add(matVecMul(Wo, combined), bo));
        double[] ct_hat = tanh(add(matVecMul(Wc, combined), bc));

        double[] newCellState = new double[cellState.length];
        for (int i = 0; i < cellState.length; i++) {
            newCellState[i] = ft[i] * cellState[i] + it[i] * ct_hat[i];
        }

        double[] newHiddenState = new double[hiddenState.length];
        for (int i = 0; i < hiddenState.length; i++) {
            newHiddenState[i] = ot[i] * Math.tanh(newCellState[i]);
        }

        double[] output = add(matVecMul(Wy, newHiddenState), by);

        double[] dOutput = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            dOutput[i] = output[i] - target[i];
        }

        double[][] dWy = new double[Wy.length][Wy[0].length];
        double[] dby = new double[by.length];
        for (int i = 0; i < Wy.length; i++) {
            for (int j = 0; j < Wy[0].length; j++) {
                dWy[i][j] = dOutput[i] * newHiddenState[j];
            }
            dby[i] = dOutput[i];
        }

        double[] dHiddenState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dHiddenState[i] += dOutput[j] * Wy[j][i];
            }
            dHiddenState[i] *= ot[i] * (1 - Math.tanh(newCellState[i]) * Math.tanh(newCellState[i]));
        }

        double[] dCellState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dCellState[i] = dHiddenState[i] * ot[i] * (1 - Math.tanh(newCellState[i]) * Math.tanh(newCellState[i]));
            dCellState[i] += newCellState[i] * (1 - newCellState[i]);
        }

        double[][] dWf = new double[Wf.length][Wf[0].length];
        double[] dbf = new double[bf.length];
        for (int i = 0; i < Wf.length; i++) {
            for (int j = 0; j < Wf[0].length; j++) {
                dWf[i][j] = dCellState[i] * cellState[i] * ft[i] * (1 - ft[i]) * combined[j];
            }
            dbf[i] = dCellState[i] * cellState[i] * ft[i] * (1 - ft[i]);
        }

        double[][] dWi = new double[Wi.length][Wi[0].length];
        double[] dbi = new double[bi.length];
        for (int i = 0; i < Wi.length; i++) {
            for (int j = 0; j < Wi[0].length; j++) {
                dWi[i][j] = dCellState[i] * it[i] * (1 - it[i]) * ct_hat[i] * combined[j];
            }
            dbi[i] = dCellState[i] * it[i] * (1 - it[i]) * ct_hat[i];
        }

        double[][] dWo = new double[Wo.length][Wo[0].length];
        double[] dbo = new double[bo.length];
        for (int i = 0; i < Wo.length; i++) {
            for (int j = 0; j < Wo[0].length; j++) {
                dWo[i][j] = dHiddenState[i] * ot[i] * (1 - ot[i]) * combined[j];
            }
            dbo[i] = dHiddenState[i] * ot[i] * (1 - ot[i]);
        }

        double[][] dWc = new double[Wc.length][Wc[0].length];
        double[] dbc = new double[bc.length];
        for (int i = 0; i < Wc.length; i++) {
            for (int j = 0; j < Wc[0].length; j++) {
                dWc[i][j] = dCellState[i] * it[i] * (1 - ct_hat[i] * ct_hat[i]) * combined[j];
            }
            dbc[i] = dCellState[i] * it[i] * (1 - ct_hat[i] * ct_hat[i]);
        }

        for (int i = 0; i < Wy.length; i++) {
            for (int j = 0; j < Wy[0].length; j++) {
                Wy[i][j] -= learningRate * dWy[i][j];
            }
            by[i] -= learningRate * dby[i];
        }

        for (int i = 0; i < Wf.length; i++) {
            for (int j = 0; j < Wf[0].length; j++) {
                Wf[i][j] -= learningRate * dWf[i][j];
            }
            bf[i] -= learningRate * dbf[i];
        }

        for (int i = 0; i < Wi.length; i++) {
            for (int j = 0; j < Wi[0].length; j++) {
                Wi[i][j] -= learningRate * dWi[i][j];
            }
            bi[i] -= learningRate * dbi[i];
        }

        for (int i = 0; i < Wo.length; i++) {
            for (int j = 0; j < Wo[0].length; j++) {
                Wo[i][j] -= learningRate * dWo[i][j];
            }
            bo[i] -= learningRate * dbo[i];
        }

        for (int i = 0; i < Wc.length; i++) {
            for (int j = 0; j < Wc[0].length; j++) {
                Wc[i][j] -= learningRate * dWc[i][j];
            }
            bc[i] -= learningRate * dbc[i];
        }

        clipGradients(clipThreshold);
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
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

    private double[][] add(double[][] a, double[][] b) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    private double[] matVecMul(double[][] a, double[] x) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = 0.0;
            for (int j = 0; j < x.length; j++) {
                result[i] += a[i][j] * x[j];
            }
        }
        return result;
    }

    // Save the model to a file
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }

    // Load the model from a file
    public static LSTMNetwork loadModel(String modelFilePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFilePath))) {
            return (LSTMNetwork) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            return null;
        }
    }

    // Get the current cell state
    public double[] getCellState() {
        return cellState;
    }

    // Get the current hidden state
    public double[] getHiddenState() {
        return hiddenState;
    }

    // Get the hidden size
    public int getHiddenSize() {
        return hiddenSize;
    }

    // Get the output weights
    public double[][] getWy() {
        return Wy;
    }

    // Get the output biases
    public double[] getBy() {
        return by;
    }
}
