# StockPricePrediction
Objective:
The goal of this project is to predict future stock prices using historical data from the stock market, specifically focusing on Amazon (AMZN) stock. This project employs two machine learning models—Support Vector Regressor (SVR) and Linear Regression—to forecast the stock's adjusted closing prices for the next 30 days.

Data Source:
The stock data is fetched using the yfinance library, which provides historical stock price information. For this project, we retrieved Amazon’s stock data for the past two months, focusing primarily on the 'Adjusted Close' price.

Methodology:
Data Collection:
Using the yfinance library, we downloaded Amazon's stock data over the past two months, including the open, high, low, close, and adjusted close prices.
Data Preprocessing:
The dataset was filtered to retain only the 'Adjusted Close' prices, as it gives a more accurate representation of the stock's value by factoring in events like dividends and stock splits.
A new column, Prediction, was created by shifting the 'Adjusted Close' prices 30 days into the future, serving as the target variable for prediction.
Feature and Target Preparation:
Features (X): The 'Adjusted Close' prices without the last 30 rows (which were reserved for prediction).
Target (Y): The Prediction column, which contains the stock prices shifted by 30 days, also excluding the last 30 rows.
Model Training:
The dataset was split into training (80%) and testing (20%) sets.
Two machine learning models were trained:
Support Vector Regressor (SVR): A powerful non-linear model with RBF kernel, which is well-suited for capturing complex relationships in the data.
Linear Regression: A linear model that assumes a linear relationship between the features and the target variable.
Model Evaluation:
Both models were evaluated based on their R² score (coefficient of determination) on the testing set to determine their predictive power.
Future Price Prediction:
The trained models were then used to forecast the stock prices for the next 30 days. These predictions provide insights into how the stock might perform in the near future.
Results:
Support Vector Regressor (SVR) Confidence: The R² score for SVR indicates how well the model fits the testing data.
Linear Regression Confidence: Similarly, the R² score for Linear Regression shows the model’s accuracy on the testing data.
Both models were used to predict the adjusted closing prices for Amazon stock over the next 30 days, and the predicted values were output for analysis.
Conclusion:
This project demonstrates the use of machine learning techniques for stock price prediction. By comparing the performance of two different models (SVR and Linear Regression), we can evaluate which method is better suited for time series forecasting of stock data. While the results provide an estimate, stock prices are influenced by a variety of unpredictable market factors, and this project should be viewed as a learning exercise in applying machine learning to financial data.

Future Work:
Explore additional features such as trading volume, technical indicators (e.g., moving averages), and external factors like market news.
Experiment with more advanced models such as Long Short-Term Memory (LSTM) networks or Gradient Boosting Machines to improve prediction accuracy.
Integrate hyperparameter tuning to optimize the performance of the SVR and Linear Regression models.
