package lstm;

import util.CustomChartUtils;
import util.DataPreprocessor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LSTMTrainer {
    private final LSTMNetwork network;
    private final double learningRate;

    public LSTMTrainer(LSTMNetwork network, double learningRate) {
        this.network = network;
        this.learningRate = learningRate;
    }

    public void train(double[][] inputs, double[][] targets, int epochs) {
        // Normalize the data
        double[][] normalizedInputs = DataPreprocessor.normalize(inputs);
        double[][] normalizedTargets = DataPreprocessor.normalize(targets);

        // Split the data into training and testing sets
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

            // Shuffle the training data
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < trainInputs.length; i++) indices.add(i);
            Collections.shuffle(indices);

            for (int i : indices) {
                double[] input = trainInputs[i];
                double[] target = trainTargets[i];

                // Reset state for each sequence
                network.resetState();

                // Forward pass
                double[] hiddenState = new double[network.getHiddenSize()];
                double[] cellState = new double[network.getHiddenSize()];
                double[] output = network.forward(input, hiddenState, cellState);
                double error = target[0] - output[0];
                totalError += error * error;

                // Backpropagation
                network.backpropagate(input, target, learningRate);
            }

            trainingLoss.add(totalError / trainInputs.length);
            validationLoss.add(validate(testInputs, testTargets));
            System.out.println("Epoch " + (epoch + 1) + " complete. Training Loss: " + totalError / trainInputs.length);
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
