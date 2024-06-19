package util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class CustomChartUtils {

    public static void saveAccuracyChart(String title, List<Integer> epochs, List<Double> accuracies, String filePath, String xLabel, String yLabel) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < epochs.size(); i++) {
            dataset.addValue((Number) accuracies.get(i), "Accuracy", epochs.get(i));
        }

        JFreeChart lineChart = ChartFactory.createLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        File lineChartFile = new File(filePath);
        ChartUtils.saveChartAsPNG(lineChartFile, lineChart, 800, 600);
    }

    public static void savePredictionChart(String title, double[] predictions, String filePath, String xLabel, String yLabel) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < predictions.length; i++) {
            dataset.addValue((Number) predictions[i], "Prediction", i);
        }

        JFreeChart lineChart = ChartFactory.createLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        File lineChartFile = new File(filePath);
        ChartUtils.saveChartAsPNG(lineChartFile, lineChart, 800, 600);
    }
}
