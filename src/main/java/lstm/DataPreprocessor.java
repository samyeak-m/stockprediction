package lstm;

import java.util.Arrays;

public class DataPreprocessor {
    public static double[][][] preprocessData(double[][] data, double splitRatio) {
        int splitIndex = (int) (data.length * splitRatio);
        double[][] trainData = Arrays.copyOfRange(data, 0, splitIndex);
        double[][] testData = Arrays.copyOfRange(data, splitIndex, data.length);

        return new double[][][]{trainData, testData};
    }
}
