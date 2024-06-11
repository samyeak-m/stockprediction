import lstm.LSTMNetwork;
import lstm.LSTMTrainer;
import database.DatabaseHelper;

import java.sql.SQLException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws SQLException {
        DatabaseHelper dbHelper = new DatabaseHelper();
        List<double[]> stockData = dbHelper.loadStockData("ntc");

        double[][] inputSequences = new double[stockData.size()][];
        double[][] targetSequences = new double[stockData.size()][];

        for (int i = 0; i < stockData.size(); i++) {
            inputSequences[i] = stockData.get(i);
            targetSequences[i] = new double[]{stockData.get(i)[1]}; // Assuming target is closing price
        }

        int inputSize = 1;
        int hiddenSize = 50;
        int outputSize = 1;
        LSTMNetwork network = new LSTMNetwork(inputSize, hiddenSize, outputSize);
        LSTMTrainer trainer = new LSTMTrainer(network, 0.001);

        trainer.train(inputSequences, targetSequences, 1000);

        // Predict next value
        double[] lastSequence = stockData.get(stockData.size() - 1);
        double[] prediction = network.forward(lastSequence, new double[hiddenSize], new double[hiddenSize]);

        // Calculate change
        double lastClosePrice = lastSequence[1];
        double predictedClosePrice = prediction[0];
        double change = predictedClosePrice - lastClosePrice;
        double pointChange = change / lastClosePrice * 100;
        String predict = pointChange > 0 ? "rise" : "fall";

        dbHelper.savePrediction("YourStockName", predict, Math.abs(pointChange), Math.abs(change), predictedClosePrice);
    }
}
