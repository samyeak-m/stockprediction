import database.DatabaseHelper;
import util.DataPreprocessor;
import lstm.LSTMNetwork;

import java.io.File;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final String MODEL_FILE_PATH = "lstm_model.ser".replace("/", File.separator);

    public static void main(String[] args) throws SQLException {
        DatabaseHelper dbHelper = new DatabaseHelper();

        LSTMNetwork lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
        if (lstm == null) {
            lstm = new LSTMNetwork(1, 50, 1); // Adjust sizes as necessary
        }

        List<String> tableNames = dbHelper.getAllStockTableNames();
        List<double[]> allStockData = new ArrayList<>();

        for (String tableName : tableNames) {
            allStockData.addAll(dbHelper.loadStockData(tableName));
        }

        double[][] stockDataArray = allStockData.toArray(new double[0][]);
        double[][][] preprocessedData = DataPreprocessor.preprocessData(stockDataArray, 0.6);
        double[][] trainData = preprocessedData[0];
        double[][] testData = preprocessedData[1];

        trainModel(lstm, trainData, 1000, 0.001);

        lstm.saveModel(MODEL_FILE_PATH);

        try (Scanner scanner = new Scanner(System.in)) {
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
        }
    }

    private static void trainModel(LSTMNetwork lstm, double[][] data, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (double[] point : data) {
                double[] input = new double[]{point[1]};
                double[] target = new double[]{point[1]}; // Assuming we're using the closing price as the target

                lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
                lstm.backpropagate(input, target, learningRate);
            }
            System.out.println("Epoch " + epoch + " completed.");
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
