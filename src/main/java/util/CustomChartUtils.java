package util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import java.io.File;
import java.io.IOException;
import java.util.List;

import java.awt.BasicStroke;
import java.awt.Color;
import javax.swing.JFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

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
                "Training Progress",
                "Epoch",
                "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false);

        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesPaint(1, Color.BLUE);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesStroke(1, new BasicStroke(2.0f));
        plot.setRenderer(renderer);

        JFrame frame = new JFrame("Training Progress");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.pack();
        frame.setVisible(true);
    }

}