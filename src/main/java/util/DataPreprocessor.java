package util;

import java.util.Map;

public class DataPreprocessor {

    public static double[][] normalize(double[][] data) {
        int columns = data[0].length;
        double[] min = new double[columns];
        double[] max = new double[columns];

        // Initialize min and max values
        for (int j = 0; j < columns; j++) {
            min[j] = Double.MAX_VALUE;
            max[j] = Double.MIN_VALUE;
        }

        // Find min and max values for each column
        for (double[] row : data) {
            for (int j = 0; j < columns; j++) {
                if (!Double.isNaN(row[j])) {
                    if (row[j] < min[j]) {
                        min[j] = row[j];
                    }
                    if (row[j] > max[j]) {
                        max[j] = row[j];
                    }
                }
            }
        }

        // Normalize data
        double[][] normalizedData = new double[data.length][columns];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < columns; j++) {
                if (!Double.isNaN(data[i][j]) && max[j] != min[j]) {
                    normalizedData[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
                } else {
                    normalizedData[i][j] = 0.0; // Or another default value
                }
            }
        }

        return normalizedData;
    }

    public static double[][][] preprocessData(double[][] data, double trainSplitRatio) {
        int trainSize = (int) (data.length * trainSplitRatio);
        int testSize = data.length - trainSize;

        double[][] trainData = new double[trainSize][data[0].length];
        double[][] testData = new double[testSize][data[0].length];

        System.arraycopy(data, 0, trainData, 0, trainSize);
        System.arraycopy(data, trainSize, testData, 0, testSize);

        return new double[][][]{trainData, testData};
    }

    public static double[][] addFeatures(double[][] stockData, double[][] technicalIndicators) {
        double[][] extendedData = new double[stockData.length][stockData[0].length + technicalIndicators[0].length];

        for (int i = 0; i < stockData.length; i++) {
            System.arraycopy(stockData[i], 0, extendedData[i], 0, stockData[i].length);
            System.arraycopy(technicalIndicators[i], 0, extendedData[i], stockData[i].length, technicalIndicators[i].length);
        }

        return extendedData;
    }

    public static double[] normalizeTableNames(Map<String, Double> tableNameMap, String[] tableNames) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        double[] normalizedValues = new double[tableNames.length];

        for (int i = 0; i < tableNames.length; i++) {
            double value = tableNameMap.get(tableNames[i]);
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
            normalizedValues[i] = value;
        }

        for (int i = 0; i < normalizedValues.length; i++) {
            normalizedValues[i] = (normalizedValues[i] - min) / (max - min);
        }

        return normalizedValues;
    }
}
