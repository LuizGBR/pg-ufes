import pandas
import numpy
import random
import os
import json
from sklearn.model_selection import StratifiedKFold

# Defining util functions

def get_data_info(json_path):
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

def get_image_files(images_path, data_csv, val_folder):
    # get train and validation indexes
    train_idx = data_csv.loc[data_csv['folder'] != val_folder].index.tolist()
    val_idx = data_csv.loc[data_csv['folder'] == val_folder].index.tolist()

    # collect training image paths
    train_paths = []
    for idx in train_idx:
        image_name = str(data_csv.loc[idx, 'img_id']) + '.png'
        if os.path.exists(os.path.join(images_path, image_name)):
            train_paths.append(os.path.join(images_path, image_name))

    # collect validation image paths
    val_paths = []
    for idx in val_idx:
        if os.path.exists(os.path.join(images_path, image_name)):
            val_paths.append(os.path.join(images_path, image_name))

    return train_paths, val_paths

def get_labels(images, label_values, csv):
    labels = list()

    for path in images:
        key = path.split('\\')[-1].split('.')[0]
        labels.append(label_values[csv.loc[csv['img_id'] == key, 'subreddit'].iloc[0]])   

    return labels

def split_k_folder_csv (data_csv, col_target, save_path=None, k_folder=5, seed_number=None):

    print("-" * 50)
    print("- Generating the {}-folders...".format(k_folder))

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pandas.read_csv(data_csv)

    skf = StratifiedKFold(n_splits=k_folder, shuffle=True, random_state=seed_number)

    target = data_csv[col_target]
    data_csv['folder'] = None

    for folder_number, (train_idx, val_idx) in enumerate(skf.split(numpy.zeros(len(target)), target)):
        data_csv.iloc[val_idx, data_csv.columns.get_loc('folder')] = folder_number + 1

    if save_path is not None:
        data_csv.to_csv(save_path, index=False)

    print("- Done!")
    print("-" * 50)

    return data_csv