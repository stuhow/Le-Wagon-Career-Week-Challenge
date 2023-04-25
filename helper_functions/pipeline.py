from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from helper_functions.preprocess import custom_imputer, time_tranformer

def pipeline(numerical_features, ordinal_features, nominal_features, time_features):
    preproc_ordinal = make_pipeline(
    custom_imputer(),
    MinMaxScaler())

    preproc_nominal = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", drop='if_binary', sparse_output=False))

    preproc_numerical = make_pipeline(
        custom_imputer(),
        MinMaxScaler(),
        # KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    )

    preproc_time = make_pipeline(
        time_tranformer())

    preproc_selector_multi = SelectFromModel(
        RandomForestClassifier(),
        threshold = "median", # drop all multivariate features lower than the median correlation
    )

    preproc_pipeline = ColumnTransformer(
        [('numerical', preproc_numerical, numerical_features),
        ('ordinal', preproc_ordinal, ordinal_features),
        ('ohe', preproc_nominal, nominal_features),
        ('time', preproc_time, time_features)],
        remainder="drop")

    final_preproc_pipeline = imbPipeline(steps=[('pipe',preproc_pipeline),
                                            ('smote',SMOTE(sampling_strategy=0.1)),
                                            ('RandomUnderSampler', RandomUnderSampler(sampling_strategy=0.5)),
                                            ('feature selecter', preproc_selector_multi),
                                            ('model', RandomForestClassifier(max_depth=4))
                                            ])
    return final_preproc_pipeline
