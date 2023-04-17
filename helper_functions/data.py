# Load data
import pandas as pd
from sklearn.model_selection import train_test_split


def load():
    # load data
    base_df = pd.read_csv('raw_data/dataset.csv', sep=';')

    # seperate na default rows for submission
    submission_dataset = base_df[base_df['default'].isnull()]
    submission_dataset.to_csv('submission_dataset.csv', index=False)

    # create base dataset without na default rows
    data = base_df[base_df['default'].notnull()]

    # set variables
    X = data.drop(columns=['default'])
    y = data.default

    # create train, test & val datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # create train, test & val dataframes
    train_dataset = X_train.join(y_train)
    test_dataset = X_test.join(y_test)
    val_dataset = X_val.join(y_val)

    # save train, test & val dataframes
    train_dataset.to_csv('raw_data/train_dataset.csv', index=False)
    test_dataset.to_csv('raw_data/test_dataset.csv', index=False)
    val_dataset.to_csv('raw_data/val_dataset.csv', index=False)
