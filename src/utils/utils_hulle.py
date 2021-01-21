import os, glob, sys
import numpy as np 
import pandas as pd 
import scipy.stats
import string
import re

def load_data(path):
    """Load training and testing datasets based on their path

    Parameters
    ----------
    path : relative path to location of data, should be always the same (string)
    
    Returns
    -------
    Training and testing Dataframes
    """
    train = pd.read_csv(os.path.join(path,'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))
    
    return train, test


def get_size_family(df, mod: bool = False):
    """Defines family relations based on the features 'SibSp' (the # of siblings / spouses aboard the Titanic)
    and 'Parch' (the # of parents / children aboard the Titanic)

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with a new feature called 'FamilySize'
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    if mod:
        
        bins_ = [0, 1, 3, 4, 5, 12]
        df['FamilySize'] = pd.cut(df["FamilySize"],  bins = bins_, labels = list(string.ascii_uppercase)[:len(bins_)-1])
        
    return df


def get_ordered_elements(df, key, order: int = 1):
    """ordered by natural order, inverted order or by survived probability
    """
    if order == 1:
        res = np.unique(df[key])
    if order == 2:
        res = np.unique(df[key])[::-1]
    elif order == 3:
        res = pd.DataFrame(df.groupby([key])['Survived'].mean()).sort_values(by='Survived', ascending=True).T.columns.values
    
    return res
    
    
def get_feature_to_numerical(df, key, elements):
    
    df[key] = df[key].apply(lambda x: list(elements).index(x))
    df[key] = df[key].astype(int)
    
    return df


def modify_fare(df, n: int = 4, extra: bool = False):
    """Introduce n new intervals (based on quantiles) for the feature fare, such that it is modified from
    being continuous to being discrete

    Parameters
    ----------
    df : panda dataframe
    n: number of new intervals (int)
    
    Returns
    -------
    Original dataframe with discretized version of the feature 'Fare', categories
    """
    #df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    
    if extra:
        
        df['Fare_num'] = df['Fare']
        
    df['Fare'] = pd.qcut(df['Fare'], n, labels = list(string.ascii_uppercase)[:n])
    
    return df


def get_title(name):
    """Search for individual title in a string by considering it to have a ASCII format from  A-Z

    Parameters
    ----------
    name : The name from which a title wants to be extracted (string)
    
    Returns
    -------
    String associated to a found title
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)

    return ""


def get_titles(df, mod: bool = True):
    """Search for all titles inside a dataframe, given the feature 'Name'

    Parameters
    ----------
    df : panda dataframe
    mod : simplify the extend of titles available (boolean)
    
    Returns
    -------
    Original dataframe with a new feature called 'Title'
    """
    df['Title'] = df['Name'].apply(get_title)
    if mod:
        # perform modifications
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    if len(p)>len(q):
        p = np.random.choice(p,len(q))
    elif len(q)>len(p):
        q = np.random.choice(q,len(p))

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def drop_features(df, to_drop):
    """Drop unwanted features 

    Parameters
    ----------
    df      : panda dataframe
    to_drop : array with name of features to be dropped
    
    Returns
    -------
    Original dataframe with all original features but those in to_drop
    """
    return df.drop(to_drop, axis=1)


def get_missing_ages(df, extra: bool = False):
    """Fills in empty Ages based on the Title of a person. If Extra is True, calculate
       the empty Ages considering the PClass and Sex.

    Parameters
    ----------
    df   : panda dataframe
    extra: number of new intervals (boolean)
    
    Returns
    -------
    Complete version the feature 'Age'
    """
    emb = []
    for i, row in df.iterrows():
        if pd.isnull(row['Age']):
            title   = row['Title']
            age_avg = df['Age'][df['Title'] == title].mean()
            age_std = df['Age'][df['Title'] == title].std()
            emb.append(np.random.randint(age_avg - age_std, age_avg + age_std, size=1)[0])
        else:
            emb.append(row['Age'])
    # Update column
    df['Age_t'] = emb
    
    if extra: 
        emb_T = []
        for i, row in df.iterrows():
            if pd.isnull(row['Age']):
                clas = row['Pclass']
                sex  = row['Sex']
                age_avg = df['Age'][(df['Pclass'] == clas) & (df['Sex'] == sex)].mean()
                age_std = df['Age'][(df['Pclass'] == clas) & (df['Sex'] == sex)].std()
                emb_T.append(np.random.randint(age_avg - age_std, age_avg + age_std, size=1)[0])
            else:
                emb_T.append(row['Age'])
        # Update column
        df['Age_ps'] = emb_T
        
    df = drop_features(df, ['Age'])

    return df


def get_ages_binned(df, key, n: int = 5):
    """Introduces n intervals for the feature given in keys, such that it is modified 
       from being continuous to be discrete

    Parameters
    ----------
    df  : panda dataframe
    key : specific Age key to modify. e.g. 'Age_t' (or could be 'Age_ps')
    n   : number of new intervals (int)
    
    Returns
    -------
    Discretized version of the feature specified by key, categories
    """

    # Create new column
    df["{}_num".format(key)] = df[key]
    
    df[key] = pd.cut(df[key], n, labels = list(string.ascii_uppercase)[:n])

    return df


def modify_titles(df):
    """Concatenates titles found to be similar or considered to be simplified in one category

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Simplified categories in the features 'Title'
    """
    # join less representative cotegories
    df['Title'] = df['Title'].replace(['Lady', 'Countess',
                                       'Capt', 'Col', 'Don', 'Dr', 'Major',
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    return df


def ticket_sep(df, mod: bool = True, red: bool = False):
    ticket_type = []

    for i in range(len(df['Ticket'])):

            ticket =df['Ticket'].iloc[i]

            for c in string.punctuation:
                ticket = ticket.replace(c,"")
                splited_ticket = ticket.split(" ")   
            if len(splited_ticket) == 1:
                ticket_type.append('NO')
            else: 
                ticket_type.append(splited_ticket[0])
                
    df["ticket_type"] = ticket_type
    
    if mod:
        for t in df['ticket_type'].unique():
            if len(df[df['ticket_type']==t]) < 15:
                df.loc[df.ticket_type ==t, 'ticket_type'] = 'other'
    if red:
        df["ticket_type"] = np.where(df["ticket_type"]==df["ticket_type"].value_counts().index[-1], df["ticket_type"].value_counts().index[-2], df["ticket_type"])
    
    return df 


def get_deck(name):
    """Search for individual Capital letter inside a string associated to the cabin of a person, from  A-Z

    Parameters
    ----------
    name : The name from which a deck wants to be extracted (string)
    
    Returns
    -------
    Letter associated with the deck from that a person has
    """    
    if pd.isnull(name):
        return 'None'
    else:
        title_search = re.findall(r"^\w", name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search[0]
        else:
            return 'None'

        
def get_decks(df, mod: bool = True):
    """Search for the information of all decks inside a dataframe, given the feature 'Cabin'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with a new feature called 'Deck'
    """
    df['Deck'] = df['Cabin'].apply(get_deck)
    # Modifications
    if mod: 
        for t in df['Deck'].unique():
            if len(df[df['Deck']==t]) <= 14:
                df.loc[df['Deck'] ==t, 'Deck'] = 'other'
        
    return df


def embarked_bayes(df, i):
    """Using Bayes Theorem, and based on 'Pclass', determine the probability of 'Embarked' for a person 
    given the possibilities S, C or Q.

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    String associated to the most likely port from where a passenger Embarked, given its Pclass
    """
    
    pclass_ = df['Pclass'].iloc[i]
    # P(s|1) = P(s)*P(1|S)/[ P(s)*P(1|s) + P(s)*P(1|s) + P(s)*P(1|s)] # probability that given the class 1, the person came from port S
    P_S, P_C, P_Q = df['Embarked'].value_counts()['S'], df['Embarked'].value_counts()['C'], \
                    df['Embarked'].value_counts()['Q']
    P_class_S = df['Embarked'][df['Pclass'] == pclass_].value_counts()['S']
    P_class_C = df['Embarked'][df['Pclass'] == pclass_].value_counts()['C']
    P_class_Q = df['Embarked'][df['Pclass'] == pclass_].value_counts()['Q']
    res = []
    P_S_class = (P_S * P_class_S) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_S_class)
    P_C_class = (P_C * P_class_C) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_C_class)
    P_Q_class = (P_Q * P_class_Q) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_C_class)

    if sorted(res, reverse=True)[0] == P_S_class:
        return 'S'
    elif sorted(res, reverse=True)[0] == P_C_class:
        return 'C'
    elif sorted(res, reverse=True)[0] == P_Q_class:
        return 'Q'

    
def get_embarked_bayes(df):
    """Search for the Embarked information of passengers missing this data, based on its 'Pclass'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with all missing values from the feature 'Embarked'
    """
    emb = []
    for i, Port in df.iterrows():
        if pd.isnull(Port['Embarked']):
            emb.append(embarked_bayes(df, i))
        else:
            emb.append(Port['Embarked'])
    # Update column
    df['Embarked'] = emb
    
    return df


def get_type_ticket(df):
    """Indicate if a person has a 'Ticket'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Categorical unique code
    """
    # Feature that tells whether a passenger had a cabin on the Titanic
    df['Type_Ticket'] = df['Ticket'].apply(lambda x: x[0:3])
    df['Type_Ticket'] = df['Type_Ticket'].astype('category').cat.codes # ordinal encoding
    df['Type_Ticket'] = df['Type_Ticket'].astype(int)
    
    return df


def get_number_cabins(df):
    
    df['number_cabins'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    
    return df


def get_extra_features(df):
    
    # Extra correlations
    df['Fare_Sex']       = df['Fare']   * df['Sex']
    df['Pclass_Sex']     = df['Pclass'] * df['Sex']
    df['Pclass_Title']   = df['Pclass'] * df['Title']
    df['Title_Sex']      = df['Title']  * df['Sex']
    df['Title_Pclass']   = df['Title']  * df['Pclass']
    df['Emb_Pclass']     = df['Pclass'] * df['Embarked']
    
    return df


def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['Survived'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    
def string_keys_used(X_train_data):
    
    return "['" +', '.join(list(X_train_data.keys())).replace(", ", "','")+ "']" 


def create_features_file(name_experiment, X_train):
    
    from datetime import datetime
    code = r'''{}'''.format(string_keys_used(X_train))
    
    f_cpp = "{}_feat_{}_{:%Y_%m_%d}.txt".format(name_experiment,len(X_train.keys()),datetime.now())
    with open(f_cpp,'w') as FOUT:
        FOUT.write(code) 
        
    return f_cpp


def save_experiment_mlflow(name_experiment, X_train, results, searchCV, searchHyp, end_run: bool = False):
    
    import mlflow
    
    mlflow.set_experiment(name_experiment)
    
    for index, classifier in results.iterrows():
        
        with mlflow.start_run():
            
            # log parameters of interest
            mlflow.log_param("model", classifier['Model'])
            # log metrics of interest
            mlflow.log_metric("cv_acc", classifier['cv_acc'])
            mlflow.log_metric("cv_acc_std", classifier['cv_acc_std'])
            if searchCV:
                mlflow.log_metric("grid_cv_acc", classifier['grid_cv_acc'])
                mlflow.log_metric("grid_cv_acc_std", classifier['grid_cv_acc_std'])
                mlflow.log_param("search_info_grid",  classifier['search_info_grid'])
                mlflow.log_param("grid_best_params",  classifier['grid_best_params'])
                mlflow.log_param("grid_time_s",  classifier['grid_time_s'])
            if searchHyp:
                mlflow.log_metric("hyper_cv_acc", classifier['hyper_cv_acc'])
                mlflow.log_metric("hyper_cv_acc_std", classifier['hyper_cv_acc_std'])
                mlflow.log_param("search_info_hyp",  classifier['search_info_hyp'])
                mlflow.log_param("hyper_best_params",  classifier['hyper_best_params'])
                mlflow.log_param("hyper_time_s",  classifier['hyper_time_s'])
            mlflow.log_param("features",  string_keys_used(X_train))
            # log artifact
            #file = create_features_file(name_experiment, X_train)
            #mlflow.log_artifact(file,artifact_path="/mnt/experiments")

    if end_run == True:
        mlflow.end_run()

    return print("Training information saved")