package lstm;

import util.CustomChartUtils;
import util.DataPreprocessor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;
import java.util.logging.Level;

public class LSTMTrainer {
    private final LSTMNetwork network;
    private final double learningRate;

    private static final Logger LOGGER = Logger.getLogger(LSTMTrainer.class.getName());


    public LSTMTrainer(LSTMNetwork network, double learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] targets, int epochs) {
        double[][] normalizedInputs = DataPreprocessor.normalize(inputs);
        double[][] normalizedTargets = DataPreprocessor.normalize(targets);

        double[][][] inputSplits = DataPreprocessor.preprocessData(normalizedInputs, 0.6);
        double[][][] targetSplits = DataPreprocessor.preprocessData(normalizedTargets, 0.6);

        double[][] trainInputs = inputSplits[0];
        double[][] testInputs = inputSplits[1];
        double[][] trainTargets = targetSplits[0];
        double[][] testTargets = targetSplits[1];

        List<Double> trainingLoss = new ArrayList<>();
        List<Double> validationLoss = new ArrayList<>();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            int correctPredictions = 0;
            long startTime = System.currentTimeMillis();

            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < trainInputs.length; i++) indices.add(i);
            Collections.shuffle(indices);

            for (int i : indices) {
                double[] input = trainInputs[i];
                double[] target = trainTargets[i];

                network.resetState();

                double[] hiddenState = new double[network.getHiddenSize()];
                double[] cellState = new double[network.getHiddenSize()];
                double[] output = network.forward(input, hiddenState, cellState);
                double error = target[0] - output[0];
                totalError += error * error;

                network.backpropagate(input, target, learningRate);

                // Calculate accuracy (if applicable)
                if (Math.abs(output[0] - target[0]) < 0.01 * target[0]) {
                    correctPredictions++;
                }
            }

            trainingLoss.add(totalError / trainInputs.length);
            validationLoss.add(validate(testInputs, testTargets));

            long endTime = System.currentTimeMillis();
            long epochTime = endTime - startTime;

            LOGGER.log(Level.INFO, String.format("Epoch %d: Accuracy = %.4f, Loss = %.6f, Time = %d ms",
                    epoch, (double) correctPredictions / trainInputs.length, totalError / trainInputs.length, epochTime));
        }

        CustomChartUtils.plotTrainingProgress(trainingLoss, validationLoss);
    }


    private double validate(double[][] inputs, double[][] targets) {
        double totalError = 0;

        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] target = targets[i];
            double[] hiddenState = new double[network.getHiddenSize()];
            double[] cellState = new double[network.getHiddenSize()];

            double[] output = network.forward(input, hiddenState, cellState);
            double error = target[0] - output[0];
            totalError += error * error;
        }

        return totalError / inputs.length;
    }
}
