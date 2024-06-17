package database;

import util.PropertyLoader;

import java.sql.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DatabaseHelper {
    private static final Logger LOGGER = Logger.getLogger(DatabaseHelper.class.getName());
    private final String url;
    private final String username;
    private final String password;

    static String RESET = "\u001B[0m";
    static String GREEN = "\u001B[32m";
    static String BLUE = "\u001B[34m";
    static String YELLOW = "\u001B[33m";

    public DatabaseHelper() {
        Properties properties = PropertyLoader.loadProperties("application.properties");
        this.url = properties.getProperty("db.url");
        this.username = properties.getProperty("db.username");
        this.password = properties.getProperty("db.password");
    }

    public Connection connect() throws SQLException {
        LOGGER.log(Level.INFO,"Connecting to the database...");
        return DriverManager.getConnection(url, username, password);
    }

    public List<String> getAllStockTableNames() throws SQLException {
        List<String> tableNames = new ArrayList<>();
        String query = "SHOW TABLES LIKE 'daily_data_%'";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query);
             ResultSet rs = pstmt.executeQuery()) {
            while (rs.next()) {
                String tableName = rs.getString(1);
                tableNames.add(tableName);
            }
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error fetching stock table names", e);
            throw e;
        }
        LOGGER.log(Level.INFO, "Fetched {0} stock table names", tableNames.size());
        return tableNames;
    }

    public List<double[]> loadStockData(String tableName) throws SQLException {
        List<double[]> stockData = new ArrayList<>();
        String query = "SELECT date, close, high, low, open, volume, turnover FROM " + tableName + " ORDER BY date";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query);
             ResultSet rs = pstmt.executeQuery()) {
            while (rs.next()) {
                java.sql.Date date = rs.getDate("date");
                double close = rs.getDouble("close");
                double high = rs.getDouble("high");
                double low = rs.getDouble("low");
                double open = rs.getDouble("open");
                double volume = rs.getDouble("volume");
                double turnover = rs.getDouble("turnover");

                double dateAsDouble = date.getTime();

                stockData.add(new double[]{dateAsDouble, close, high, low, open, volume, turnover});
            }
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error loading stock data for table " + tableName, e);
            throw e;
        }
        LOGGER.log(Level.INFO, "Loaded {0} rows of stock data from table {1}", new Object[]{stockData.size(), tableName});
        return stockData;
    }

    private void createPredictionsTableIfNotExists() throws SQLException {
        String createTableSQL = "CREATE TABLE IF NOT EXISTS predictions (" +
                "id INT AUTO_INCREMENT PRIMARY KEY, " +
                "stock_symbol VARCHAR(10) NOT NULL, " +
                "predict VARCHAR(255) NOT NULL, " +
                "point_change VARCHAR(255) NOT NULL, " +
                "price_change DOUBLE NOT NULL, " +
                "prediction DOUBLE NOT NULL, " +
                "actual DOUBLE NOT NULL, " +
                "date DATE NOT NULL, " +
                "prediction_date DATE NOT NULL" +
                ")";

        try (Connection conn = connect();
             Statement stmt = conn.createStatement()) {
            stmt.executeUpdate(createTableSQL);
            LOGGER.log(Level.INFO, "Ensured that predictions table exists");
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error creating predictions table", e);
            throw e;
        }
    }

    public void savePrediction(String stockSymbol, String predict, String pointChangeStr, double priceChange, double prediction, double actual, LocalDate today, LocalDate predictionDate) throws SQLException {
        createPredictionsTableIfNotExists();

        String query = "INSERT INTO predictions (stock_symbol, predict, point_change, price_change, prediction, actual, date, prediction_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";

        try (Connection conn = connect();
             PreparedStatement pstmt = conn.prepareStatement(query)) {
            pstmt.setString(1, stockSymbol);
            pstmt.setString(2, predict);
            pstmt.setString(3, pointChangeStr);
            pstmt.setDouble(4, priceChange);
            pstmt.setDouble(5, prediction);
            pstmt.setDouble(6, actual);
            pstmt.setDate(7, Date.valueOf(today));
            pstmt.setDate(8, Date.valueOf(predictionDate));
            pstmt.executeUpdate();
            LOGGER.log(Level.INFO, "Saved prediction for stock {0}", stockSymbol);
        } catch (SQLException e) {
            LOGGER.log(Level.SEVERE, "Error saving prediction for stock " + stockSymbol, e);
            throw e;
        }
    }
}
