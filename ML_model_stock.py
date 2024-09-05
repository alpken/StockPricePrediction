import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import date
import dateutil.relativedelta

# Set date range for fetching data
end_date = date.today()
start_date = end_date - dateutil.relativedelta.relativedelta(months=2)

# Download stock data for Amazon (AMZN)
df = yf.download("AMZN", start=start_date, end=end_date)
df.reset_index(inplace=True)

# Display the first few rows of the dataframe
print(df.head())

# Focus on 'Adj Close' column
df = df[['Adj Close']]
print(df.head())

# Number of days out for future prediction
forecast_out = 30

# Create a new column 'Prediction' shifted by 30 days into the future
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
print(df.head())

# Create the feature dataset (X) and convert to NumPy array
X = np.array(df.drop(['Prediction'], axis=1))  # axis=1 specifies column-wise drop
X = X[:-forecast_out]  # Remove the last 30 rows corresponding to NaN predictions

# Create the target dataset (Y) and convert to NumPy array
Y = np.array(df['Prediction'])
Y = Y[:-forecast_out]  # Remove the last 30 rows from Y as well

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create and train the Support Vector Regressor model (SVR) with RBF kernel
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Test the SVR model confidence (R^2 score) on the test set
svm_confidence = svr_rbf.score(x_test, y_test)
print("SVR confidence: ", svm_confidence)

# Create and train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Test the Linear Regression model confidence (R^2 score) on the test set
lr_confidence = lr.score(x_test, y_test)
print("Linear Regression confidence: ", lr_confidence)

# Prepare the forecast data (the last 30 rows of 'Adj Close' values)
x_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]
print(x_forecast)

# Make predictions using the trained Linear Regression model
lr_prediction = lr.predict(x_forecast)
print("Linear Regression predictions: ", lr_prediction)

# Make predictions using the trained SVR model
svm_prediction = svr_rbf.predict(x_forecast)
print("SVR predictions: ", svm_prediction)
