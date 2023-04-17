# preprocess, train & predict
from helper_functions.data import load

def load_split_data():
    '''
    A function that loads the raw data, breaks out the submission dataset and
    then spliet the rest into train, test, validation datasets
    '''
    load()


def preprocess():

    # train_dataset = pd.read_csv('train_dataset.csv')

    # X_train = train_dataset.drop(columns=['default'])
    # y_train = train_dataset.default

    # categorical_features = ['account_status',
    #                     'account_worst_status_0_3m',
    #                     'account_worst_status_12_24m',
    #                     'account_worst_status_3_6m',
    #                     'account_worst_status_6_12m',
    #                     'merchant_category',
    #                     'merchant_group',
    #                     'has_paid',
    #                     'name_in_email',
    #                     'status_last_archived_0_24m',
    #                     'status_2nd_last_archived_0_24m',
    #                     'status_3rd_last_archived_0_24m',
    #                     'status_max_archived_0_6_months',
    #                     'status_max_archived_0_12_months',
    #                     'status_max_archived_0_24_months',
    #                     'worst_status_active_inv']

    # numerical_features = [x for x in X_train.columns if x not in categorical_features]

    pass

def train():
    pass

def evaluate():
    pass

def predict():
    pass

if __name__ == "__main__":
    load_split_data()
    preprocess()
    train()
    evaluate()
    predict()
