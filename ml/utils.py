import pandas
import random
import os

# Defining util functions

def get_data_info(info_path):
    # Read the JSON file into a pandas DataFrame
    df = pandas.read_json(info_path)

    # Melt the DataFrame to combine all columns into a single column named 'value'
    csv = pandas.melt(df, var_name='img_id', value_name='value')

    # Adding a new column with the column index (starting at 1)
    csv['col'] = 'value_' + (csv.groupby('img_id').cumcount() + 1).astype(str)

    # Pivoting the DataFrame
    csv = csv.pivot(index='img_id', columns='col', values='value').reset_index()

    # Renaming the columns
    csv.columns = ['img_id', 'subreddit', 'audit', 'is_removed']

    # Set the index to the 'img_id' column
    csv = csv.set_index('img_id')

    return csv

def get_image_groups(path, train_percent):
    # Get a list of all the image file names in the directory
    image_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

    # Calculate the number of images to use for training and testing
    num_train = int(len(image_files) * train_percent)
    num_test = len(image_files) - num_train

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Split the shuffled list of image files into two separate lists
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]

    return train_files, test_files

def get_labels(images, label_values, csv):
    labels = list()

    for path in images:
        key = path.split('\\')[-1].split('.')[0]
        labels.append(label_values[csv.subreddit[key]])   

    return labels