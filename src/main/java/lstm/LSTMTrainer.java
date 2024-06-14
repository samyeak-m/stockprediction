package lstm;

public class LSTMTrainer {
    private final LSTMNetwork network;
    private final double learningRate;

    public LSTMTrainer(LSTMNetwork network, double learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];
                double[] hiddenState = new double[network.getHiddenSize()];
                double[] cellState = new double[network.getHiddenSize()];

                double[] output = network.forward(input, hiddenState, cellState);
                double error = target[0] - output[0];
                double gradient = error * learningRate;

                double[][] dWy = new double[network.getWy().length][network.getWy()[0].length];
                double[] dby = new double[network.getBy().length];

                for (int j = 0; j < dWy.length; j++) {
                    for (int k = 0; k < dWy[j].length; k++) {
                        dWy[j][k] = gradient * hiddenState[k];
                    }
                    dby[j] = gradient;
                }

                updateWeights(network.getWy(), dWy);
                updateBiases(network.getBy(), dby);
            }
        }
    }

    private void updateWeights(double[][] weights, double[][] gradients) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] += gradients[i][j];
            }
        }
    }

    private void updateBiases(double[] biases, double[] gradients) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] += gradients[i];
        }
    }
}
