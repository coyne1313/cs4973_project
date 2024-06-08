'''
Cosine similarity model for use in final deliverable
Author: Seamus Coyne
Date: Jun. 8 2024
'''

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def find_closest_country(merged_df, user_pref, top_n):
    '''
    Function: find_closest_country()
    Params:
        merged_df: Dataframe containing all countries data (one line per year)
        user_pref: Dictionary of user preference inputs
        top_n: Number of closest matches to be used
    Returns: top_n closest matches; defaults to 5
    '''

    # Define the features to scale
    feats = merged_df.columns[1:-1]
    X = merged_df[feats]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert scaled features to DataFrame and add 'country' column
    X_scaled_df = pd.DataFrame(X_scaled, columns=feats)
    X_scaled_df['country'] = merged_df['country']
    X_scaled_df = X_scaled_df.dropna()

    # Create DataFrame from user preferences
    user_df = pd.DataFrame([user_pref])

    # Scale user preferences using the same scaler
    user_scaled = scaler.transform(user_df)

    # Calculate cosine similarity
    sim = cosine_similarity(user_scaled, X_scaled_df.drop(columns=['country']))
    top_matches = sim[0].argsort()[-top_n:][::-1]

    # Find the matching country
    match_countries = X_scaled_df.iloc[top_matches]['country'].values

    # Output the result
    return [f"match_{i + 1}: {country.title()}" for i, country in
            enumerate(match_countries)]