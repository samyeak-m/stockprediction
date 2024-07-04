import util.CustomChartUtils;
import util.DataPreprocessor;
import util.TechnicalIndicators;
import lstm.LSTMNetwork;
import lstm.LSTMTrainer;
import database.DatabaseHelper;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Main {
    static String version = "v1";

    private static final String MODEL_FILE_PATH = "lstm_model" + version + ".ser";
    private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String BLUE = "\u001B[34m";
    private static final String YELLOW = "\u001B[33m";

    static int hiddenSize = 100;
    static int inputSize = 9;
    static int outputSize = 1;
    static int epoch = 15;
    static double trainingRate = 0.001;

    private static final List<Integer> epochList = new ArrayList<>();
    private static final List<Double> accuracyList = new ArrayList<>();

    public static void main(String[] args) throws SQLException, IOException, ClassNotFoundException {
        LOGGER.setLevel(Level.INFO);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.INFO);
        LOGGER.addHandler(handler);

        DatabaseHelper dbHelper = new DatabaseHelper();

        LSTMNetwork lstm = LSTMNetwork.loadModel(MODEL_FILE_PATH);
        if (lstm == null) {
            lstm = new LSTMNetwork(inputSize, hiddenSize, outputSize);
        }

        List<String> tableNames = dbHelper.getAllStockTableNames();
        List<double[]> allStockData = new ArrayList<>();

        for (String tableName : tableNames) {
            allStockData.addAll(dbHelper.loadStockData(tableName));
        }

        double[][] stockDataArray = allStockData.toArray(new double[0][]);

        // Calculate technical indicators
        double[][] technicalIndicators = TechnicalIndicators.calculate(stockDataArray, 20, 20);

        // Combine stock data with technical indicators
        double[][] extendedData = DataPreprocessor.addFeatures(stockDataArray, technicalIndicators);

        double[][][] preprocessedData = DataPreprocessor.preprocessData(extendedData, 0.6);
        double[][] trainData = preprocessedData[0];
        double[][] testData = preprocessedData[1];


        LOGGER.log(Level.INFO, BLUE + "Training data size: " + trainData.length + RESET);
        LOGGER.log(Level.INFO, BLUE + "Test data size: " + testData.length + RESET);

        trainModel(lstm, trainData, epoch, trainingRate);

        double accuracy = testModel(lstm, testData);

        lstm.saveModel(MODEL_FILE_PATH);

        String accuracyChartDir = "charts" + version + File.separator + "accuracy";
        String predictionChartDir = "charts" + version + File.separator + "predictions";
        createDirectory(accuracyChartDir);
        createDirectory(predictionChartDir);

        CustomChartUtils.saveAccuracyChart("Model Accuracy", epochList, accuracyList, accuracyChartDir + File.separator + "model_accuracy.png", "Epochs", "Accuracy");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("Enter the stock symbol to predict: ");
                String stockSymbol = scanner.nextLine();
                System.out.print("Enter the number of days for prediction: ");
                int days = scanner.nextInt();
                scanner.nextLine();

                predictAndSave(dbHelper, lstm, stockSymbol, days, predictionChartDir);

                System.out.print(BLUE + "Do you want to predict for another stock? (yes/no): " + RESET);
                String response = scanner.nextLine();
                if (!response.equalsIgnoreCase("yes")) {
                    break;
                }
            }
        }
    }

    private static void trainModel(LSTMNetwork lstm, double[][] trainData, int epochs, double learningRate) {
        LSTMTrainer trainer = new LSTMTrainer(lstm, learningRate);
        double prevAccuracy = 0;
        int sameCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            int totalDataPoints = trainData.length;
            int batchSize = 32; // Example batch size, adjust as needed
            int batches = totalDataPoints / batchSize;

            // Shuffle trainData
            shuffleArray(trainData);

            double totalLoss = 0;

            // Train in batches
            for (int batch = 0; batch < batches; batch++) {
                double[][] batchData = Arrays.copyOfRange(trainData, batch * batchSize, (batch + 1) * batchSize);
                for (double[] data : batchData) {
                    double[] input = Arrays.copyOfRange(data, 0, data.length - 1);
                    double[] target = new double[]{data[data.length - 1]};
                    lstm.backpropagate(input, target, learningRate);

                    double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());

                    if (output == null) {
                        LOGGER.severe("NaN value encountered during forward pass. Stopping training.");
                        return; // Or implement other error handling logic
                    }
                }
            }

            // Calculate accuracy and loss
            double accuracy = testModel(lstm, trainData);
            double epochLoss = calculateLoss(lstm, trainData);

            // Logging
            long endTime = System.currentTimeMillis();
            long elapsedTime = endTime - startTime;

            epochList.add(epoch);
            accuracyList.add(accuracy);

            LOGGER.log(Level.INFO, String.format(YELLOW + "Epoch %d: Accuracy = %.4f, Loss = %.6f, Time = %d ms" + RESET, epoch, accuracy, epochLoss, elapsedTime));

            // Check if accuracy is the same as previous epoch
            if (Math.abs(accuracy - prevAccuracy) < 0.01) {
                sameCount++;
            } else {
                sameCount = 0;
            }

            // If accuracy is the same for 2 consecutive epochs, reinitialize the model
            if (sameCount == 2) {
                lstm = new LSTMNetwork(inputSize, hiddenSize, outputSize);
                trainer = new LSTMTrainer(lstm, learningRate);
                sameCount = 0;
            }

            prevAccuracy = accuracy;
        }
    }

    private static double testModel(LSTMNetwork lstm, double[][] testData) {
        int correctPredictions = 0;
        for (double[] point : testData) {
            double[] input = new double[point.length - 1];
            System.arraycopy(point, 0, input, 0, point.length - 1);

            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            if (output == null) {
                continue; // Skip this iteration if output is null
            }
            double prediction = output[0];
            double actual = point[point.length - 1];

            System.out.println("actual price     : " +actual);
            System.out.println("prediction price : "+prediction);

            if (Math.abs(prediction - actual) < 0.01 * actual) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / testData.length;
    }

    private static double calculateLoss(LSTMNetwork lstm, double[][] data) {
        double totalLoss = 0;
        for (double[] point : data) {
            double[] input = new double[point.length - 1];
            System.arraycopy(point, 0, input, 0, point.length - 1);
            double[] output = lstm.forward(input, lstm.getHiddenState(), lstm.getCellState());
            double prediction = output[0];
            double actual = point[point.length - 1];
            totalLoss += Math.pow(actual - prediction, 2); // MSE loss
        }
        return totalLoss / data.length;
    }

    private static void shuffleArray(double[][] array) {
        List<double[]> list = Arrays.asList(array);
        Collections.shuffle(list);
        list.toArray(array);
    }

    private static void createDirectory(String directory) {
        File dir = new File(directory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    private static void predictAndSave(DatabaseHelper dbHelper, LSTMNetwork lstm, String stockSymbol, int days, String predictionChartDir) throws SQLException, IOException {
        List<double[]> stockData = dbHelper.loadStockData(stockSymbol);
        double[][] stockDataArray = stockData.toArray(new double[0][]);
        double[][] technicalIndicators = TechnicalIndicators.calculate(stockDataArray, 20, 20);
        double[][] extendedData = DataPreprocessor.addFeatures(stockDataArray, technicalIndicators);

        double[][] input = new double[days][extendedData[0].length];
        for (int i = 0; i < days; i++) {
            System.arraycopy(extendedData[i], 0, input[i], 0, extendedData[i].length);
        }

        double[] predictions = new double[days];
        for (int i = 0; i < days; i++) {
            double[] currentInput = Arrays.copyOfRange(input[i], 0, input[i].length - 1);
            double[] output = lstm.forward(currentInput, lstm.getHiddenState(), lstm.getCellState());
            predictions[i] = output[0];
        }

        CustomChartUtils.savePredictionChart("Predictions for " + stockSymbol, predictions, predictionChartDir + File.separator + stockSymbol + "_predictions.png", "Days", "Price");
    }
}
