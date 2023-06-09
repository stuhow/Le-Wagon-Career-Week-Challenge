# preprocess, train & predict
from helper_functions.data import load
from helper_functions.pipeline import pipeline
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score



def load_split_data():
    '''
    A function that loads the raw data, breaks out the submission dataset and
    then spliet the rest into train, test, validation datasets
    '''
    load()


def train(numerical_features, ordinal_features, nominal_features, time_features):
    # import training dataset
    train_dataset = pd.read_csv('raw_data/train_dataset.csv')

    # set target and variables
    X_train = train_dataset.drop(columns=['default'])
    y_train = train_dataset.default

    # preprocessing pipeline
    preproc_pipeline = pipeline(numerical_features, ordinal_features, nominal_features, time_features)

    # fit to training data
    preproc_pipeline.fit(X_train, y_train)

    # Export Pipeline as pickle file
    with open("models/pipeline.pkl", "wb") as file:
        pickle.dump(preproc_pipeline, file)

    pass

def evaluate():

    # Load Pipeline from pickle file
    my_pipeline = pickle.load(open("models/pipeline.pkl","rb"))

    # import testing dataset
    train_dataset = pd.read_csv('raw_data/train_dataset.csv')

    # set target and variables
    X_test = train_dataset.drop(columns=['default'])
    y_test = train_dataset.default

    print(my_pipeline.get_params())

    # cross validate
    # cv_score = cross_val_score(my_pipeline, X_test, y_test, cv=5, scoring='accuracy')
    # print(cv_score.mean())
    # pass

def predict():
    pass

numerical_features = ['account_amount_added_12_24m',
                      'account_days_in_dc_12_24m',
                      'account_days_in_rem_12_24m',
                      'account_days_in_term_12_24m',
                      'account_incoming_debt_vs_paid_0_24m',
                      'age',
                      'avg_payment_span_0_12m',
                      'avg_payment_span_0_3m',
                      'max_paid_inv_0_12m',
                      'max_paid_inv_0_24m',
                      'num_active_div_by_paid_inv_0_12m',
                      'num_active_inv',
                      'num_arch_dc_0_12m',
                      'num_arch_dc_12_24m',
                      'num_arch_ok_0_12m',
                      'num_arch_ok_12_24m',
                      'num_arch_rem_0_12m',
                      'num_arch_written_off_0_12m',
                      'num_arch_written_off_12_24m',
                      'num_unpaid_bills',
                      'recovery_debt',
                      'sum_capital_paid_account_0_12m',
                      'sum_capital_paid_account_12_24m',
                      'sum_paid_inv_0_12m']

ordinal_features = ['account_status',
                    'account_worst_status_0_3m',
                    'account_worst_status_12_24m',
                    'account_worst_status_3_6m',
                    'account_worst_status_6_12m',
                    'status_last_archived_0_24m',
                    'status_2nd_last_archived_0_24m',
                    'status_3rd_last_archived_0_24m',
                    'status_max_archived_0_6_months',
                    'status_max_archived_0_12_months',
                    'status_max_archived_0_24_months',
                    'worst_status_active_inv']

nominal_features = ['merchant_category', 'merchant_group', 'has_paid', 'name_in_email']

time_features = ['time_hours']

if __name__ == "__main__":
    # load_split_data()
    # train(numerical_features, ordinal_features, nominal_features, time_features)
    evaluate()
    predict()
