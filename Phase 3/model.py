import pandas as pd
import numpy as np

countries = "Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, Sweden"
eu_countries = countries.split(", ")

df_crime_input = pd.read_csv("Phase 3/Data/crime_training.csv")

def add_bias_column(X):
    """
    Args:
        X (array): can be either 1-d or 2-d
    
    Returns:
        Xnew (array): the same array, but 2-d with a column of 1's in the first spot
    """
    
    # If the array is 1-d
    if len(X.shape) == 1:
        Xnew = np.column_stack([np.ones(X.shape[0]), X])
    
    # If the array is 2-d
    elif len(X.shape) == 2:
        bias_col = np.ones((X.shape[0], 1))
        Xnew = np.hstack([bias_col, X])
        
    else:
        raise ValueError("Input array must be either 1-d or 2-d")

    return Xnew


def line_of_best_fit(X, y):
    X = add_bias_column(X)
    return np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))


def train(df_crime):
    """

    trains time series model based on crime per year per country

    Args:
       crime_df (dataframe) - dataframe with three columns: country, year, and amount of crime per 100k people
    
    Returns:
       regression (array) - array representing intercept and slopes of linear regression
    
    """

    df_crime_scaled = pd.DataFrame()

    df_crime_scaled["country"] = df_crime["country"]
    df_crime_scaled["year"] = df_crime["year"]
    df_crime_scaled["amount"] = (df_crime["amount"] - df_crime["amount"].mean()) / df_crime["amount"].std()


    df_crime_dummies = df_crime_scaled.join(pd.get_dummies(df_crime_scaled["country"]))
    
    crime_arr = np.array(df_crime_dummies.drop(columns=["country"]))

    
    

    X_left = crime_arr[:, 0:1]
    X_right = crime_arr[:, 2:]

    

    X = np.hstack((X_left, X_right))
    

    y = crime_arr[:, 1:2]
    
    regression = line_of_best_fit(X, y)
    return regression


def predict (country, year, regression, mean, std):
    """
    following linear regression model, predicts the total amount of crimes per 100k people 

    Args:
       - country (int) - which country wanted for prediction
       - year (int) - which year we wanted for prediction
       - regression (array) - linear regression intercept and slopes (scaled)
       - mean (int) - mean of original data
       - std (int) - std of original data

    Returns:
       - answer (int) - predicted value for crimes per 100k people 
    
    """
    

    encoding = [0] * len(eu_countries)

    encoding[eu_countries.index(country)] = 1

    encoding.insert(0, year)
    encoding.insert(0, 1)

    print(encoding)

    unscaled_answer = np.matmul(encoding, regression)[0]

    answer = unscaled_answer * std + mean




    

    return answer
print(train(df_crime_input))

#print(predict("Sweden", 2025, train(crime_df)))
#print(predict("Romania", 2025, train(df_crime_input), df_crime_input["amount"].mean(), df_crime_input["amount"].std()) )
#print(crime_df["amount"])

#print(crime_df["amount"].mean())
#print(crime_df["amount"].std())

#print(crime_df.describe())
print(df_crime_input)







