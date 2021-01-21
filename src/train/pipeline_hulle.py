import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from src.utils import utils_hulle 

def transformation_features(df, age_extra: bool = False, fare_extra: bool = False):
    """get new or modified features based on EDA

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Dataframe with all studied modifications
    """
    df = utils_hulle.get_size_family(df, mod = True) # returns cat
    df = utils_hulle.get_feature_to_numerical(df, 'FamilySize',
                                              utils_hulle.get_ordered_elements(df,'FamilySize', 1)) # cat to num
    df = utils_hulle.modify_fare(df, 4, fare_extra) # returns cat, True: cat & num
    df = utils_hulle.get_feature_to_numerical(df, 'Fare', 
                                              utils_hulle.get_ordered_elements(df,'Fare', 1)) # cat to num
    df = utils_hulle.get_titles(df, True) # returns categories
    df = utils_hulle.get_missing_ages(df, age_extra) # numerical # True = numerical x 2
    df = utils_hulle.get_ages_binned(df, 'Age_t', 5) # returns categories
    df = utils_hulle.get_feature_to_numerical(df, 'Age_t', 
                                              utils_hulle.get_ordered_elements(df,'Age_t', 1)) # cat to num
    if age_extra:
        df = utils_hulle.get_ages_binned(df, 'Age_ps', 5) # returns categories
        df = utils_hulle.get_feature_to_numerical(df, 'Age_ps', 
                                                  utils_hulle.get_ordered_elements(df,'Age_ps', 1)) # cat to num
    df = utils_hulle.modify_titles(df) # returns categories
    df = utils_hulle.get_feature_to_numerical(df, 'Title', 
                                              utils_hulle.get_ordered_elements(df,'Title')) # cat to num # hot
    df = utils_hulle.ticket_sep(df, True, True) # returns categories
    df = utils_hulle.get_feature_to_numerical(df, 'ticket_type',
                                              utils_hulle.get_ordered_elements(df,'ticket_type')) # cat to num # hot
    df = utils_hulle.get_decks(df, True) # returns categories
    df = utils_hulle.get_feature_to_numerical(df, 'Deck', 
                                              utils_hulle.get_ordered_elements(df,'Deck', 2)) # cat to num
    df = utils_hulle.get_embarked_bayes(df) # returns categories
    df = utils_hulle.get_feature_to_numerical(df, 'Embarked', 
                                              utils_hulle.get_ordered_elements(df,'Embarked')) # cat to num # hot
    df = utils_hulle.get_number_cabins(df) # numerical
    df = utils_hulle.get_feature_to_numerical(df, 'Sex',
                                              utils_hulle.get_ordered_elements(df,'Sex')) # cat to num # hot

    return df


def pipeline_features(df, cat_hot_features, unchanged, to_scale, scale_op: int = 1):
    
    # Define categorical pipeline
    cat_hot_pipe = Pipeline([('encoder', OneHotEncoder(sparse=False))])
    
    # Define categorical pipeline
    unchanged_pipe = Pipeline([('other_encoding', None)])
    
    if  scale_op  == 1:
        scaler = StandardScaler() 
    elif scale_op == 2:
        scaler = RobustScaler()
    elif scale_op == 3: 
        scaler = MinMaxScaler()
    
    # Define categorical pipeline
    scaling_pipe = Pipeline([('scaling', scaler)])
    
    # Fit column transformer to training data
    preprocessor = ColumnTransformer(transformers=[
                                                   ('cat',      cat_hot_pipe, cat_hot_features),
                                                   ('none',     unchanged_pipe,   unchanged),
                                                   ('scaling',  scaling_pipe, to_scale)])
    preprocessor.fit(df)

    # Prepare column names
    if len(cat_hot_features) != 0:
        cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names(cat_hot_features)
        columns = np.hstack(( cat_columns, unchanged, to_scale )).ravel()
    else:
        columns = np.hstack(( unchanged, to_scale )).ravel()
    
    return pd.DataFrame(preprocessor.transform(df), columns=columns)