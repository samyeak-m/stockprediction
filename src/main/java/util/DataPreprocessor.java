package util;

public class DataPreprocessor {

    // Adds technical indicators to stock data
    public static double[][] addFeatures(double[][] stockData, double[][] technicalIndicators) {
        int numRows = stockData.length;
        int numCols = stockData[0].length + technicalIndicators[0].length;

        double[][] extendedData = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            System.arraycopy(stockData[i], 0, extendedData[i], 0, stockData[i].length);
            System.arraycopy(technicalIndicators[i], 0, extendedData[i], stockData[i].length, technicalIndicators[i].length);
        }

        return extendedData;
    }

    // Splits data into training and testing sets based on the trainSplitRatio
    public static double[][][] preprocessData(double[][] data, double trainSplitRatio) {
        int trainSize = (int) (data.length * trainSplitRatio);
        double[][] trainData = new double[trainSize][];
        double[][] testData = new double[data.length - trainSize][];

        System.arraycopy(data, 0, trainData, 0, trainSize);
        System.arraycopy(data, trainSize, testData, 0, data.length - trainSize);

        return new double[][][]{trainData, testData};
    }

    // Normalizes data to have zero mean and unit variance
    public static double[][] normalize(double[][] data) {
        int numRows = data.length;
        int numCols = data[0].length;
        double[][] normalizedData = new double[numRows][numCols];

        for (int j = 0; j < numCols; j++) {
            double mean = 0;
            double std = 0;

            // Calculate mean
            for (int i = 0; i < numRows; i++) {
                mean += data[i][j];
            }
            mean /= numRows;

            // Calculate standard deviation
            for (int i = 0; i < numRows; i++) {
                std += Math.pow(data[i][j] - mean, 2);
            }
            std = Math.sqrt(std / numRows);

            // Normalize the data
            for (int i = 0; i < numRows; i++) {
                if (std == 0) {
                    normalizedData[i][j] = 0;  // Prevent division by zero
                } else {
                    normalizedData[i][j] = (data[i][j] - mean) / std;
                }
            }
        }

        return normalizedData;
    }
}
