'''
File: context_linreg.py
Content: Phase II Linear Regression Model transferred from .ipynb filetype
Author: Seamus Coyne
Jun. 5 2024
'''

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyjstat import pyjstat
from sklearn.preprocessing import MinMaxScaler

eu_nations = ['austria', 'belgium', 'bulgaria', 'croatia', 'cyprus', 'czechia', 'denmark',
              'estonia', 'finland', 'france', 'germany', 'greece', 'hungary', 'ireland',
              'italy', 'latvia', 'lithuania', 'luxembourg', 'malta', 'netherlands', 'poland',
              'portugal', 'romania', 'slovakia', 'spain', 'sweden', 'slovenia']

eu_nations = [nation.upper() for nation in eu_nations]

def get_eurostat_data(article_title):
    '''
    Fetch Eurostat data by article title.
    '''
    article_title = article_title.upper()
    url = f'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{article_title}?format=JSON&lang=EN'
    output = pyjstat.Dataset.read(url)
    df = output.write('dataframe')
    return df

# Fetch crime data
crime = get_eurostat_data('crim_off_cat')
crime.dropna(inplace=True)
crime['Time'] = crime['Time'].astype(int)
crime['Geopolitical entity (reporting)'] = crime['Geopolitical entity (reporting)'].str.upper()
crime = crime[crime['Geopolitical entity (reporting)'].isin(eu_nations)]

scaler = MinMaxScaler()
crime['norm_val'] = scaler.fit_transform(crime[['value']])

# Assume crime data is already loaded into a DataFrame called 'crime'
robbery_data = crime[crime['International classification of crime for statistical purposes (ICCS)'] == 'Robbery']


# Define the linear regression function
def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    return m, b


# Add function to predict y values
def linreg_predict(X, y, coefficients):
    m, b = coefficients
    ypreds = m * X + b
    resids = y - ypreds
    return {'ypreds': ypreds, 'resids': resids}


# Function to plot residuals and predictions
def plot_residuals_predictions(ypreds, resids):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(ypreds, resids)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')

    plt.subplot(1, 2, 2)
    plt.plot(resids, marker='o', linestyle='none')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Index Order')

    plt.tight_layout()
    plt.show()


# Loop over countries and perform linear regression
countries = robbery_data['Geopolitical entity (reporting)'].unique()
for country in countries:
    country_data = robbery_data[
        robbery_data['Geopolitical entity (reporting)'] == country]

    x = country_data['Time'].values
    y = country_data['norm_val'].values

    # Perform linear regression
    coefficients = linear_regression(x, y)

    # Calculate R-squared value
    ypreds = linreg_predict(x, y, coefficients)['ypreds']
    ss_res = np.sum((y - ypreds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot regression line using Plotly
    m, b = coefficients
    regression_line = m * x + b

    fig = px.scatter(country_data, x='Time', y='norm_val',
                     title=f'Robbery Cases in {country} Over Time')

    fig.add_traces(go.Scatter(x=x, y=regression_line, mode='lines',
                              name='Regression Line'))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Cases (Scaled)',
        template='plotly_white',
        width=900,
        height=600,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.update_xaxes(tickangle=45)

    # Add R-squared value as text annotation
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.95, y=0.05,
        text=f'R-squared: {r_squared:.2f}',
        showarrow=False,
        font=dict(size=12, color='black'),
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='rgba(0, 0, 0, 0.7)',
        borderwidth=1
    )

    fig.show()

    # Plot residuals and predictions for diagnostic checks
    plot_residuals_predictions(ypreds, y - ypreds)

print(crime)
