package database;

import util.PropertyLoader;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class DatabaseHelper {
    private final String url;
    private final String username;
    private final String password;

    public DatabaseHelper() {
        Properties properties = PropertyLoader.loadProperties("application.properties");
        this.url = properties.getProperty("db.url");
        this.username = properties.getProperty("db.username");
        this.password = properties.getProperty("db.password");
    }

    public Connection connect() throws SQLException {
        return DriverManager.getConnection(url, username, password);
    }

    public List<double[]> loadStockData(String stockName) throws SQLException {
        List<double[]> stockData = new ArrayList<>();
        String query = "SELECT * FROM daily_data WHERE symbol = ? ORDER BY date";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query)) {
            pstmt.setString(1, stockName);
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    double[] data = new double[2]; // Assuming you have two columns, e.g., date and close price
                    data[0] = rs.getDate("date").getTime();
                    data[1] = rs.getDouble("close");
                    stockData.add(data);
                }
            }
        }
        return stockData;
    }

    public void savePrediction(String stockName, String predict, double pointChange, double priceChange, double finalPrice) throws SQLException {
        String createTableQuery = "CREATE TABLE IF NOT EXISTS nepse_price (" +
                "symbol VARCHAR(50), " +
                "predict VARCHAR(10), " +
                "point DOUBLE, " +
                "price DOUBLE, " +
                "final_price DOUBLE" +
                ")";
        String insertQuery = "INSERT INTO nepse_price (symbol, predict, point, price, final_price) VALUES (?, ?, ?, ?, ?)";

        try (Connection conn = connect();
             Statement stmt = conn.createStatement()) {
            stmt.execute(createTableQuery);
        }

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(insertQuery)) {
            pstmt.setString(1, stockName);
            pstmt.setString(2, predict);
            pstmt.setDouble(3, pointChange);
            pstmt.setDouble(4, priceChange);
            pstmt.setDouble(5, finalPrice);
            pstmt.executeUpdate();
        }
    }
}
