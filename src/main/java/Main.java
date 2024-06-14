
import database.DatabaseHelper;
import lstm.DataPreprocessor;
import lstm.LSTMNetwork;

import java.io.IOException;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final String MODEL_FILE_PATH = "lstm_model.ser";

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
        DatabaseHelper dbHelper = new DatabaseHelper();
        LSTMNetwork lstm;

        // Load or create the model
        try {
            lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
            System.out.println("Model loaded from file.");
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("No existing model found. Creating a new one.");
            lstm = new LSTMNetwork(1, 50, 1); // Adjust sizes as necessary
        }

        // Load data from all stock tables
        List<String> tableNames = dbHelper.getAllStockTableNames();
        List<double[]> allStockData = new ArrayList<>();

        for (String tableName : tableNames) {
            allStockData.addAll(dbHelper.loadStockData(tableName));
        }

        // Convert list to array for processing
        double[][] stockDataArray = allStockData.toArray(new double[0][]);

        // Preprocess data
        double[][][] preprocessedData = DataPreprocessor.preprocessData(stockDataArray, 0.6);
        double[][] trainData = preprocessedData[0];
        double[][] testData = preprocessedData[1];

        // Train the model
        trainModel(lstm, trainData, 1000, 0.001);

        // Save the model
        lstm.saveModel(MODEL_FILE_PATH);

        // Predict
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Enter the stock symbol to predict: ");
            String stockSymbol = scanner.nextLine();
            System.out.print("Enter the number of days for prediction: ");
            int days = scanner.nextInt();
            scanner.nextLine();  // Consume newline

            predictAndSave(dbHelper, lstm, stockSymbol, days);

            System.out.print("Do you want to predict for another stock? (yes/no): ");
            String response = scanner.nextLine();
            if (!response.equalsIgnoreCase("yes")) {
                break;
            }
        }

        scanner.close();
    }

    private static void trainModel(LSTMNetwork lstm, double[][] data, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] hiddenState = new double[lstm.getHiddenSize()];
            double[] cellState = new double[lstm.getHiddenSize()];

            for (double[] point : data) {
                double[] input = new double[]{point[1]};
                double[] target = new double[]{point[1]}; // Assuming we're using the closing price as the target
                double[] output = lstm.forward(input, hiddenState, cellState);

                // Compute loss (mean squared error)
                double loss = 0;
                for (int i = 0; i < output.length; i++) {
                    loss += Math.pow(output[i] - target[i], 2);
                }
                loss /= output.length;

                // Backpropagation and weights update (simplified)
                // Implement proper backpropagation and weights update logic here

                System.out.println("Epoch " + epoch + ", Loss: " + loss);
            }
        }
    }

    private static void predictAndSave(DatabaseHelper dbHelper, LSTMNetwork lstm, String stockSymbol, int days) throws SQLException {
        List<double[]> stockData = dbHelper.loadStockData("daily_data_" + stockSymbol);

        double[] hiddenState = new double[lstm.getHiddenSize()];
        double[] cellState = new double[lstm.getHiddenSize()];

        for (int i = 0; i < days; i++) {
            double[] input = new double[]{stockData.get(stockData.size() - 1)[1]};
            double[] output = lstm.forward(input, hiddenState, cellState);

            double prediction = output[0];
            stockData.add(new double[]{stockData.get(stockData.size() - 1)[0] + 86400000, prediction});

            LocalDate today = LocalDate.now();
            LocalDate predictionDate = today.plusDays(i + 1);
            String predict = (prediction - input[0]) / input[0] > 0.1 ? "up" : (prediction - input[0]) / input[0] < -0.1 ? "down" : "neutral";
            double priceChange = prediction - input[0];
            String pointChangeStr = String.format("%.2f", (priceChange / input[0]) * 100);

            if (Double.parseDouble(pointChangeStr) > 10) {
                pointChangeStr = "10";
                prediction = input[0] * 1.10;
            } else if (Double.parseDouble(pointChangeStr) < -10) {
                pointChangeStr = "-10";
                prediction = input[0] * 0.90;
            }

            dbHelper.savePrediction(stockSymbol, predict, pointChangeStr, priceChange, prediction, input[0], today, predictionDate);

            System.out.println("Predicted next closing price for " + stockSymbol + " on " + predictionDate + ": " + prediction);
            System.out.println("Prediction saved successfully.");
        }
    }
}
