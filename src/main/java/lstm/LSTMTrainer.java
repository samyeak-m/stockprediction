package lstm;

import java.util.Arrays;
import java.util.Random;

public class LSTMTrainer {
    private final LSTMNetwork lstm;
    private final double learningRate;

    public LSTMTrainer(LSTMNetwork lstm, double learningRate) {
        this.lstm = lstm;
        this.learningRate = learningRate;
    }

    public void train(double[][] trainData, int epochs, int batchSize) {
        int totalDataPoints = trainData.length;
        int batches = totalDataPoints / batchSize;

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffleArray(trainData);

            for (int batch = 0; batch < batches; batch++) {
                double[][] batchData = getBatch(trainData, batch, batchSize);
                for (double[] data : batchData) {
                    double[] input = getInput(data);
                    double[] target = getTarget(data);
                    lstm.backpropagate(input, target, learningRate);
                }
            }
        }
    }

    private double[][] getBatch(double[][] data, int batchIndex, int batchSize) {
        int start = batchIndex * batchSize;
        int end = Math.min(start + batchSize, data.length);
        double[][] batch = new double[end - start][];
        System.arraycopy(data, start, batch, 0, end - start);
        return batch;
    }

    private double[] getInput(double[] data) {
        return Arrays.copyOfRange(data, 0, data.length - 1);
    }

    private double[] getTarget(double[] data) {
        return new double[]{data[data.length - 1]};
    }

    private void shuffleArray(double[][] array) {
        Random rand = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
