package lstm;

import java.util.Random;

class LSTMCell {
    double[] Wf, Wi, Wc, Wo; // Weight matrices for forget, input, cell, and output gates
    double[] Uf, Ui, Uc, Uo; // Weight matrices for recurrent connections
    double[] bf, bi, bc, bo; // Bias vectors

    double[] h, c; // Hidden state and cell state

    public LSTMCell(int inputSize, int hiddenSize) {
        Wf = new double[hiddenSize * inputSize];
        Wi = new double[hiddenSize * inputSize];
        Wc = new double[hiddenSize * inputSize];
        Wo = new double[hiddenSize * inputSize];

        Uf = new double[hiddenSize * hiddenSize];
        Ui = new double[hiddenSize * hiddenSize];
        Uc = new double[hiddenSize * hiddenSize];
        Uo = new double[hiddenSize * hiddenSize];

        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bc = new double[hiddenSize];
        bo = new double[hiddenSize];

        h = new double[hiddenSize];
        c = new double[hiddenSize];

        // Initialize weights and biases (e.g., with random values)
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < Wf.length; i++) Wf[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Wi.length; i++) Wi[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Wc.length; i++) Wc[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Wo.length; i++) Wo[i] = rand.nextGaussian() * 0.1;

        for (int i = 0; i < Uf.length; i++) Uf[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Ui.length; i++) Ui[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Uc.length; i++) Uc[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < Uo.length; i++) Uo[i] = rand.nextGaussian() * 0.1;

        for (int i = 0; i < bf.length; i++) bf[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < bi.length; i++) bi[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < bc.length; i++) bc[i] = rand.nextGaussian() * 0.1;
        for (int i = 0; i < bo.length; i++) bo[i] = rand.nextGaussian() * 0.1;
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

    public double[] forward(double[] x) {
        // Calculate the gates
        double[] f = sigmoid(add(dot(Wf, x), dot(Uf, h), bf));
        double[] i = sigmoid(add(dot(Wi, x), dot(Ui, h), bi));
        double[] c_tilde = tanh(add(dot(Wc, x), dot(Uc, h), bc));
        double[] o = sigmoid(add(dot(Wo, x), dot(Uo, h), bo));

        // Update cell state and hidden state
        for (int j = 0; j < c.length; j++) {
            c[j] = f[j] * c[j] + i[j] * c_tilde[j];
            h[j] = o[j] * Math.tanh(c[j]);
        }

        return h;
    }

    private double[] dot(double[] W, double[] x) {
        // Implement the dot product of W and x
        double[] result = new double[W.length / x.length];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < x.length; j++) {
                result[i] += W[i * x.length + j] * x[j];
            }
        }
        return result;
    }

    private double[] add(double[] a, double[] b, double[] c) {
        // Add three vectors element-wise
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i] + c[i];
        }
        return result;
    }
}
