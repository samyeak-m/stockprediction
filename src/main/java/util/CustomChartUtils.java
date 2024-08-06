package util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class CustomChartUtils {

    public static void plotTrainingProgress(List<Double> trainingLoss, List<Double> validationLoss) {
        XYSeries trainingSeries = new XYSeries("Training Loss");
        XYSeries validationSeries = new XYSeries("Validation Loss");

        for (int i = 0; i < trainingLoss.size(); i++) {
            trainingSeries.add(i + 1, trainingLoss.get(i));
            validationSeries.add(i + 1, validationLoss.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(trainingSeries);
        dataset.addSeries(validationSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Training and Validation Loss",
                "Epoch",
                "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Optional: Customize the chart appearance here (e.g., colors, gridlines)
        chart.setBackgroundPaint(java.awt.Color.white);

        // Save or display the chart
        try {
            ChartUtils.saveChartAsPNG(new File("training_progress.png"), chart, 800, 600);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveAccuracyChart(String title, List<Integer> epochs, List<Double> accuracies, String filePath, String xLabel, String yLabel) throws IOException {
        XYSeries accuracySeries = new XYSeries("Accuracy");

        for (int i = 0; i < epochs.size(); i++) {
            accuracySeries.add(epochs.get(i), accuracies.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(accuracySeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Optional: Customize the chart appearance here
        chart.setBackgroundPaint(java.awt.Color.white);

        // Save the chart to a file
        ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }

    public static void saveLossChart(String title, List<Integer> epochs, List<Double> losses, String filePath, String xLabel, String yLabel) throws IOException {
        XYSeries lossSeries = new XYSeries("Loss");

        for (int i = 0; i < epochs.size(); i++) {
            lossSeries.add(epochs.get(i), losses.get(i));
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(lossSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                yLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Optional: Customize the chart appearance here
        chart.setBackgroundPaint(java.awt.Color.white);

        // Save the chart to a file
        ChartUtils.saveChartAsPNG(new File(filePath), chart, 800, 600);
    }
}
