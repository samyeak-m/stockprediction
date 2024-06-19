package util;

import java.util.ArrayList;
import java.util.List;

public class DataPreprocessor {
    public static double[][] addFeatures(double[][] data) {
        List<double[]> extendedData = new ArrayList<>();

        for (int i = 0; i < data.length; i++) {
            double[] row = data[i];
            double[] newRow = new double[row.length + 2];  // Adding space for moving average and RSI
            System.arraycopy(row, 0, newRow, 0, row.length);

            // Calculate moving average (e.g., 10-day)
            if (i >= 9) {
                double sum = 0;
                for (int j = i - 9; j <= i; j++) {
                    sum += data[j][1];  // Assuming the close price is at index 4
                }
                newRow[row.length] = sum / 10;
            } else {
                newRow[row.length] = row[1];  // Default to current close price if not enough data
            }

            // Calculate RSI (e.g., 14-day)
            if (i >= 14) {
                double gain = 0, loss = 0;
                for (int j = i - 13; j <= i; j++) {
                    double change = data[j][1] - data[j - 1][1];
                    if (change > 0) {
                        gain += change;
                    } else {
                        loss -= change;
                    }
                }
                double rs = gain / loss;
                newRow[row.length + 1] = 100 - (100 / (1 + rs));
            } else {
                newRow[row.length + 1] = 50;  // Default RSI to 50 if not enough data
            }

            extendedData.add(newRow);
        }

        return extendedData.toArray(new double[0][]);
    }

    public static double[][][] preprocessData(double[][] data, double splitRatio) {
        int trainSize = (int) (data.length * splitRatio);
        double[][] trainData = new double[trainSize][];
        double[][] testData = new double[data.length - trainSize][];

        System.arraycopy(data, 0, trainData, 0, trainSize);
        System.arraycopy(data, trainSize, testData, 0, data.length - trainSize);

        return new double[][][]{trainData, testData};
    }
}
