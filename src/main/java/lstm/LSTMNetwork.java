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
            newHiddenState[i] = ot[i] * tanh(newCellState[i]);
        }

        double[] output = add(matVecMul(Wy, newHiddenState), by);

        System.arraycopy(newHiddenState, 0, this.hiddenState, 0, hiddenSize);
        System.arraycopy(newCellState, 0, this.cellState, 0, hiddenSize);

        return output;
    }

    private double[] matVecMul(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private double[] add(double[] vec1, double[] vec2) {
        double[] result = new double[vec1.length];
        for (int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] + vec2[i];
        }
        return result;
    }

    private double[] sigmoid(double[] vec) {
        double[] result = new double[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = 1 / (1 + Math.exp(-vec[i]));
        }
        return result;
    }

    private double[] tanh(double[] vec) {
        double[] result = new double[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = Math.tanh(vec[i]);
        }
        return result;
    }

    // New method to apply tanh on a single double value
    private double tanh(double value) {
        return Math.tanh(value);
    }

    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }

    public static LSTMNetwork loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (LSTMNetwork) ois.readObject();
        }
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public double[][] getWf() { return Wf; }
    public double[][] getWi() { return Wi; }
    public double[][] getWo() { return Wo; }
    public double[][] getWc() { return Wc; }
    public double[][] getWy() { return Wy; }
    public double[] getBf() { return bf; }
    public double[] getBi() { return bi; }
    public double[] getBo() { return bo; }
    public double[] getBc() { return bc; }
    public double[] getBy() { return by; }

    public void setWf(double[][] Wf) { System.arraycopy(Wf, 0, this.Wf, 0, Wf.length); }
    public void setWi(double[][] Wi) { System.arraycopy(Wi, 0, this.Wi, 0, Wi.length); }
    public void setWo(double[][] Wo) { System.arraycopy(Wo, 0, this.Wo, 0, Wo.length); }
    public void setWc(double[][] Wc) { System.arraycopy(Wc, 0, this.Wc, 0, Wc.length); }
    public void setWy(double[][] Wy) { System.arraycopy(Wy, 0, this.Wy, 0, Wy.length); }
    public void setBf(double[] bf) { System.arraycopy(bf, 0, this.bf, 0, bf.length); }
    public void setBi(double[] bi) { System.arraycopy(bi, 0, this.bi, 0, bi.length); }
    public void setBo(double[] bo) { System.arraycopy(bo, 0, this.bo, 0, bo.length); }
    public void setBc(double[] bc) { System.arraycopy(bc, 0, this.bc, 0, bc.length); }
    public void setBy(double[] by) { System.arraycopy(by, 0, this.by, 0, by.length); }
}
