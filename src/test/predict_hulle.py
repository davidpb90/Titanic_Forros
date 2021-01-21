import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlflow

from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials 
from sklearn.tree import DecisionTreeClassifier # Generic Decision Tree
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.svm import SVC # Support Vector machine
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_searchCV(model, data, score: str = 'accuracy', search_op: str = 'random', 
                   n_iter_: int = 10, cv_:int = 5):
    import time
    # get data
    X_train, Y_train = data    
    # bookkepping results
    results = {}
    if search_op == 'random':
        cl = RandomizedSearchCV(model['Model'], model['hyp_parm'],
                                cv=StratifiedKFold(n_splits=cv_), n_jobs=-1, n_iter=n_iter_, scoring=score)
    else:
        cl = GridSearchCV(model['Model'], model['hyp_parm'], 
                          cv=StratifiedKFold(n_splits=cv_), n_jobs=-1, scoring=score)
        
    start = time.time()  # Get start time
    # Fit the learner to the training data using 
    grid_search = cl.fit(X_train, Y_train)
    end = time.time()  # Get end time
    #print(classification_report(Y_test, Y_pred_t, output_dict=False))
    # Save best parameters
    results['grid_cv_acc'] = np.max(grid_search.cv_results_['mean_test_score'])
    results['grid_cv_acc_std'] = grid_search.cv_results_['std_test_score'][np.argmax(grid_search.cv_results_['mean_test_score'])]*2
    results['grid_best_params'] = grid_search.best_params_
    # Calculate the training time
    results['grid_time_s'] = end - start
    
    return results

def train_hyperopt(model_name, space, data, score: str = 'accuracy', eval_: int = 200, cv_:int = 5):
    
    import time
    # get data
    X_train, Y_train = data
    results = {}
    
    if model_name == 'RF':
        
        space = space[model_name]['hyperopt']
        
        def objective_RF(space):

            model = RandomForestClassifier( criterion = space['criterion'],
                                            max_depth = space['max_depth'],
                                            max_features = space['max_features'],
                                            min_samples_leaf = space['min_samples_leaf'],
                                            min_samples_split = space['min_samples_split'],
                                            bootstrap = space['bootstrap'],
                                            oob_score = space['oob_score'], 
                                            n_estimators = space['n_estimators'],
                                            random_state=42
                                            )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_RF, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = RandomForestClassifier(**best_params,n_jobs=-1)
        
    elif model_name == 'KN':
        
        space = space[model_name]['hyperopt']
        
        def objective_KN(space):
            
            model = KNeighborsClassifier( n_neighbors = space['n_neighbors'],
                                    weights = space['weights'],
                                    algorithm = space['algorithm'],
                                    leaf_size = space['leaf_size'],
                                    p = space['p'],
                                    metric = space['metric']
                                    )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_KN, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = KNeighborsClassifier(**best_params,n_jobs=-1)
        
    elif model_name == 'LR':
        
        space = space[model_name]['hyperopt']
        
        def objective_KN(space):
            
            model = LogisticRegression( solver = space['solver'],
                                        penalty = space['penalty'],
                                        dual = space['dual'],
                                        C = space['C'],
                                        class_weight = space['class_weight'],
                                        intercept_scaling = space['intercept_scaling'],
                                        max_iter = space['max_iter'],
                                        fit_intercept = space['fit_intercept'],
                                        tol = space['tol'],
                                        warm_start = space['warm_start'],
                                        random_state=42
                                        )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_KN, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = LogisticRegression(**best_params,n_jobs=-1)
        
    elif model_name == 'DT':
        
        space = space[model_name]['hyperopt']
        
        def objective_DT(space):

            model = DecisionTreeClassifier( min_samples_split = space['min_samples_split'],
                                            min_samples_leaf = space['min_samples_leaf'],
                                            criterion = space['criterion'],
                                            max_features = space['max_features'],
                                            max_depth = space['max_depth'],
                                            random_state=42
                                            )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_DT, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = DecisionTreeClassifier(**best_params)
        
    elif model_name == 'SGD':
        
        space = space[model_name]['hyperopt']
        
        def objective_SGD(space):
            
            model = SGDClassifier( loss = space['loss'],
                                    penalty = space['penalty'],
                                    max_iter = space['max_iter'],
                                    alpha = space['alpha'],
                                    l1_ratio = space['l1_ratio'],
                                    average = space['average'],
                                    random_state=42
                                    )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_SGD, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = SGDClassifier(**best_params)
        
    elif model_name == 'LGBM':
        
        space = space[model_name]['hyperopt']
        
        def objective_LGBM(space):

            model = LGBMClassifier( 
                                    boosting_type = space['boosting_type'],
                                    max_depth = space['max_depth'],
                                    num_leaves = space['num_leaves'],
                                    learning_rate = space['learning_rate'],
                                    subsample_for_bin = space['subsample_for_bin'],
                                    colsample_bytree = space['colsample_bytree'],
                                    min_child_samples = space['min_child_samples'],
                                    reg_alpha= space['reg_alpha'],
                                    reg_lambda= space['reg_lambda'], 
                                    importance_type = space['importance_type'], 
                                    n_estimators = space['n_estimators'],
                                    random_state=42
                                  )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_LGBM, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = LGBMClassifier(**best_params,n_jobs=-1)
        
    elif model_name == 'XGB':
        
        space = space[model_name]['hyperopt']
        
        def objective_XGB(space):
            model = XGBClassifier(  
                                    max_depth = space['max_depth'],
                                    learning_rate = space['learning_rate'],
                                    subsample = space['subsample'],
                                    colsample_bytree = space['colsample_bytree'],
                                    colsample_bylevel = space['colsample_bylevel'],
                                    min_child_weight= space['min_child_weight'],
                                    gamma= space['gamma'], 
                                    reg_lambda = space['reg_lambda'], 
                                    n_estimators = space['n_estimators'],
                                    random_state=42
                                  )
            metric = cross_val_score(model, X_train, Y_train, 
                                     cv=StratifiedKFold(n_splits=cv_),scoring=score,n_jobs=-1).mean()
            # We aim to maximize accuracy, therefore we return it as a negative value
            return {'loss': -metric, 'status': STATUS_OK }
        # Evaluate
        start = time.time()  # Get start time
        trials = Trials()
        best = fmin(fn= objective_XGB, space= space, 
                    algo= tpe.suggest, max_evals = eval_, trials= trials)
        end = time.time()  # Get end time
        # Calculate the training time
        results['hyper_time_s'] = end - start
        # Best parameters
        best_params = space_eval(space, best)
        # Save best parameters
        results['hyper_best_params'] = best_params
        model = XGBClassifier(**best_params,n_jobs=-1)
        
    cv_res = cross_val_score(model, X_train, Y_train, 
                             cv=StratifiedKFold(n_splits=5),scoring = score)

    results['hyper_cv_acc']      = cv_res.mean()
    results['hyper_cv_acc_std']  = cv_res.std()
        
    return results

def prediction_overview(list_models, data, searchCV, CVOP, searchHyp, HypOp):
    
    if searchCV:
        grid_op, n_iter_  = CVOP
    if searchHyp:
        eval_, cv_        = HypOp
        
    X_train, Y_train = data
    df_results = pd.DataFrame.from_dict({})
    
    for name, model in tqdm(list_models.items()):
        print("Training and predicting on: {}".format(name))
        res = {}
        res['Model'] = "{}".format(name)
        # Compute CV
        cv_r  = cross_val_score(model['Model'], X_train, Y_train, cv=StratifiedKFold(n_splits=5), scoring = 'accuracy')
        res['cv_acc']      = cv_r.mean()
        res['cv_acc_std']  = cv_r.std()*2
        # If desired, compute and add GridSearch optimization
        if searchCV:
            if grid_op == 'random':
                res['search_info_grid']  = "{}_n_{}".format(grid_op, n_iter_)
            else:
                res['search_info_grid']  = "{}".format(grid_op)
            res.update(train_searchCV(model, data, 'accuracy', grid_op, n_iter_))
        if searchHyp:
            res['search_info_hyp']  = "hyp_{}_ev".format(eval_)
            res.update(train_hyperopt(name, list_models, data, 'accuracy', eval_, cv_))
        # save results in dataframe
        df_results = pd.concat([df_results,pd.DataFrame.from_dict(res,orient='index').T],ignore_index=True)
        
    return df_results

def make_prediction(model, X_train_data, Y_train, X_test):
    """Add predictions from a specific model to the dataframe (test)

    Parameters
    ----------
    model                 : model considered for fitting training data (object)
    X_train_data, Y_train : array with training features and target labels
    frame                 : testing frame where to add predictions from trained model 
    
    Returns
    -------
    Original dataframe with new feature 'Survived', calculated by using a specific model
    """
    
    model.fit(X_train_data, Y_train)
    
    predictions = model.predict(X_test.drop(columns = ['PassengerId', 'Survived']))
    
    frame = pd.DataFrame(data={'PassengerId': X_test['PassengerId'].astype(int), 'Survived': predictions.astype(int)})
    
    return frame


def make_prediction_opt(prediction, X_test):
    """Add predictions from a specific model to the dataframe (test)

    Parameters
    ----------
    Prediction
    frame                 : testing frame where to add predictions from trained model 
    
    Returns
    -------
    Original dataframe with new feature 'Survived', calculated by using a specific model
    """
    
    frame = pd.DataFrame(data={'PassengerId': X_test['PassengerId'].astype(int), 'Survived': prediction.astype(int)})
    
    return frame


def save_experiment_mlflow(name_experiment, results, top_3: bool = True, end_run: bool = False):
        
    mlflow.set_experiment(name_experiment)
    
    if top_3:
        res = results.sort_values(by=['cv_acc'], ascending=False)[:3]
    
    for index, classifier in res.iterrows():
        
        with mlflow.start_run():
            
            # log parameters of interest
            mlflow.log_param("classifier", classifier['Model'])
            mlflow.log_param("best_params",  classifier['best_params'])
            # log metrics of interest
            mlflow.log_metric("cv_acc", classifier['cv_acc'])
            mlflow.log_metric("cv_acc_std", classifier['cv_acc_std'])

    if end_run == True:
        mlflow.end_run()

    return print("Training information saved")