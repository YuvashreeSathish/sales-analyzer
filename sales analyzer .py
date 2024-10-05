import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv("supermarket_sales - Sheet1.csv")
df.head()
missing_values = df.isnull().sum()
missing_values
df.dtypes
df.columns
df.describe()
df['cost_goods'] = df['cogs'] - df['Tax 5%']
df['profit'] = df['Revenue'] - df['cost_goods']

if not (df['profit'] > 0).all():
  
  plt.figure(figsize=(10, 6))
  plt.stackplot(df.index, df['cogs'], df['Revenue'], df['profit'], labels=['COGS', 'Revenue', 'Profit'], colors=['lightblue', 'yellow', 'red'])
  plt.title('Cumulative Area Plot of COGS, Revenue, and Profit')
  plt.xlabel('Index')
  plt.ylabel('Amount')
  plt.legend(loc='upper left')
  plt.grid(True)
  plt.show()
else:
  plt.figure(figsize=(10, 6))
  plt.stackplot(df.index, df['cost_goods'], df['Revenue'], df['profit'], labels=['COGS', 'Revenue', 'Profit'], colors=['lightblue', 'yellow', 'red'])
  plt.title('Cumulative Area Plot of COGS, Revenue, and Profit')
  plt.xlabel('Index')
  plt.ylabel('Amount')
  plt.legend(loc='upper left')
  plt.grid(True)
  plt.show()
df_clustering = df[['Customer type', 'Gender', 'Revenue']]
df_clustering = pd.get_dummies(df_clustering)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_clustering)
predicted_clusters = kmeans.predict(df_clustering)
df_clustering['Segment'] = predicted_clusters
df_segment_sales = df_clustering.groupby('Segment')['Revenue'].mean()
print(df_segment_sales)
duplicate_invoices = df[df.duplicated(['Invoice ID'])]
print(duplicate_invoices[['Invoice ID', 'Branch', 'City', 'Product line', 'Unit price', 'Quantity', 'Revenue', 'Date', 'Time', 'Payment']])
df['z_score'] = (df['Revenue'] - df['Revenue'].mean()) / df['Revenue'].std()
outliers = df[(df['z_score'] > 3) | (df['z_score'] < -3)]
print(outliers[['Invoice ID', 'Branch', 'City', 'Product line', 'Unit price', 'Quantity', 'Revenue', 'Date', 'Time', 'Payment']])
df_segment_type = df_clustering.groupby(['Segment', 'Customer type_Member', 'Customer type_Normal']).size()
print(df_segment_type)
df_segment_gender = df_clustering.groupby(['Segment', 'Gender_Female', 'Gender_Male']).size()
print(df_segment_gender)
df_classification = df[['Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Revenue', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating']]
df_classification = pd.get_dummies(df_classification)
df_classification['Rating'] = pd.cut(df_classification['Rating'], bins=3, labels=[1, 2, 3])
X_train, X_test, y_train, y_test = train_test_split(df_classification.drop('Rating', axis=1), df_classification['Rating'], test_size=0.2)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predicted_ratings = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
print('Accuracy:', accuracy)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_ratings)
print('Mean absolute error:', mae)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predicted_ratings)
print('Mean squared error:', mse)
from sklearn.metrics import mean_squared_log_error
rmsle = mean_squared_log_error(y_test, predicted_ratings)
print('Root mean squared error:', rmsle)
df_new = df[['Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Revenue', 'Payment', 'cogs', 'gross margin percentage', 'gross income']]
df_new = pd.get_dummies(df_new)
predicted_ratings = rf.predict(df_new)
print(predicted_ratings)
df_regression = df[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Revenue']]
df_regression = pd.get_dummies(df_regression)
X_train, X_test, y_train, y_test = train_test_split(df_regression.drop('Revenue', axis=1), df_regression['Revenue'], test_size=0.2)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predicted_totals = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
print('Accuracy:', accuracy)
df_new = df[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Payment', 'cogs', 'gross margin percentage', 'gross income']]
df_new = pd.get_dummies(df_new)
predicted_totals = rf.predict(df_new)
print(predicted_totals)
# Convert 'Date' column to datetime if it's not already in datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Set 'Date' column as index
df.set_index('Date', inplace=True)
# Resample data to monthly frequency if needed
df_monthly = df.resample('M').sum()
# Define the SARIMA parameters (p, d, q, P, D, Q, s) for both non-seasonal and seasonal components
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12  # Assuming monthly data with yearly seasonality
model_sarima = SARIMAX(df_monthly['Revenue'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_results = model_sarima.fit()
# Forecast future data
forecast_steps = 12  # Forecasting for the next 12 months
sarima_forecast = sarima_results.forecast(steps=forecast_steps)
# Plot historical data and forecasted values
plt.plot(df_monthly.index, df_monthly['Revenue'], label='Historical Data')
plt.plot(pd.date_range(start=df_monthly.index[-1], periods=forecast_steps, freq='M'), sarima_forecast, label='Forecasted Data')
plt.title('Historical Revenue and Forecasted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
import torch
def generate_sales_suggestion(prompt, max_length=100):
    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length,
        temperature=0.8,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        top_k=0
    )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)
if not (df['profit'] > 0).all():
  prompt="\n How can we adjust pricing strategies to mitigate loss value without sacrificing competitiveness in supermarket?give 3 solutions"
  print(generate_sales_suggestion(prompt))
else:
  prompt = "\n Which operational areas offer the greatest potential for profit value improvement in supermarket?.give 3 solutions"
  print(generate_sales_suggestion(prompt))