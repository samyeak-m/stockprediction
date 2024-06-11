package lstm;

public class LSTMTrainer {
    private LSTMNetwork network;
    private double learningRate;

    public LSTMTrainer(LSTMNetwork network, double learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputSequences, double[][] targetSequences, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;

            for (int i = 0; i < inputSequences.length; i++) {
                double[] input = inputSequences[i];
                double[] target = targetSequences[i];
                double[] output = network.forward(input, network.getHiddenState(), network.getCellState());

                // Calculate error (mean squared error)
                double error = 0;
                for (int j = 0; j < output.length; j++) {
                    error += Math.pow(output[j] - target[j], 2);
                }
                error /= output.length;
                totalError += error;

                // Backpropagation and weight update
                network.backward(input, target, output, learningRate);
            }

            System.out.println("Epoch " + epoch + ", Error: " + totalError);
        }
    }
}