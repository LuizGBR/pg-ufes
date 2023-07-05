import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH, REDDIT_BASE_PATH
sys.path.insert(0, RAUG_PATH)
import pandas
import os
import json
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number

def json_to_csv(json_path):
    # Read the JSON file into a Python object
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert the Python object into a DataFrame
    df = pandas.DataFrame(data.values(), index=data.keys())

    # Reset the index to a column and rename it
    df = df.reset_index().rename(columns={'index': 'img_id'})

    # Reorder the columns
    df = df[['img_id', 'subreddit', 'audit', 'is_removed']]

    return df

import pandas as pd

def keep_rows_with_values(dataframe, column, values):
    """
    Keeps only the rows in the DataFrame where the specified column has any of the specified values.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The column to filter.
        values (list): A list of values to keep in the specified column.
    
    Returns:
        pd.DataFrame: The DataFrame with only the rows where the specified column has any of the specified values.
    """
    filtered_df = dataframe[dataframe[column].isin(values)]
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


data_csv = json_to_csv(os.path.join(REDDIT_BASE_PATH, "dataset.json"))

filtered_data_csv = keep_rows_with_values(data_csv, column="subreddit", values=["Dermatology", "skincancer", "skin", "Skincare_Addiction", "Rosacea", "Psoriasis",
                                                                                "SkincareAddiction", "30PlusSkinCare", "Warts", "peeling", "Acne", "eczema",
                                                                                "popping", "SkincareAddicts"])

data = split_k_folder_csv(filtered_data_csv, "subreddit", save_path=None, k_folder=6, seed_number=13)

data_test = data[ data['folder'] == 6]
data_train = data[ data['folder'] != 6]

data_test.to_csv(os.path.join(REDDIT_BASE_PATH, "reddit_parsed_test.csv"), index=False)
label_categorical_to_number (os.path.join(REDDIT_BASE_PATH, "reddit_parsed_test.csv"), "subreddit",
                             col_target_number="subreddit_number",
                             save_path=os.path.join(REDDIT_BASE_PATH, "reddit_parsed_test.csv"))

data_train = data_train.reset_index(drop=True)
data_train = split_k_folder_csv(data_train, "subreddit", save_path=None, k_folder=5, seed_number=13)
label_categorical_to_number (data_train, "subreddit", col_target_number="subreddit_number",
                             save_path=os.path.join(REDDIT_BASE_PATH, "reddit_parsed_folders.csv"))





