from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, \
BaggingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf


PATH_DATA = Path('.')


def plot_img(data, nrows=3, ncols=3, data_rows=28, data_cols=28, \
    figsize_x=9, figsize_y=9, random_seed=10):

    np.random.seed(random_seed)

    fig, ax = plt.subplots(ncols, nrows, figsize=(figsize_x, figsize_y))
    ax = ax.ravel()

    rnd_idx = np.random.randint(0, len(data), size=nrows*ncols)

    data_reshaped = np.array(data).reshape(-1, data_rows, data_cols, 1)
    for i, idx in enumerate(rnd_idx):
        ax[i].imshow(data_reshaped[idx])

    return fig, ax




def create_subfile_dr(test_data, pca, model, name_model, PATH_DATA=PATH_DATA):
    """
    Creates a submission file ready for kaggle
    
    Parameters
    ------------
    test_data: pandas dataframe, the test set;
    pca: sklearn pca object, fit to the training set;
    model: sklearn model, fit to the training set;
    name_model: string, name of the sklearn model;
    PATH_DATA: pathlib object, relative path to create the submission file to.
    
    Returns:
    - 0 if executed correctly;
    - 1 if pca transformation of the test set went wrong.
    """

    ids = test_data.index.values + 1
    try:
        test_data_tr = pca.transform(test_data)
    except:
        print('PCA transformation of test set gone wrong')
        return 1

    model_loaded = load_model(name_model)
    if (model_loaded == None):
        predictions = model.predict(test_data_tr)
    else:
        predictions = model_loaded.predict(test_data_tr)


    """
    BIT PRESENT IN THE OLD VERSION AND NOW DEPRECATED
    # try:
    #     model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
    #     predictions = model_loaded.predict(test_data_tr)
    # except FileNotFoundError:
    #     print('model not stored in' / PATH_DATA)
    #     predictions = model.predict(test_data_tr)
    """


    sub_file = pd.DataFrame({'ImageId': ids, 'Label': predictions})
    sub_file.to_csv(PATH_DATA / f'submission_{name_model}.csv', sep=',', index=False)

    return 0



def save_model(model, name_model, overwrite='no', PATH_DATA=PATH_DATA):
    """
    Saves a sklearn model

    Parameters
    ------------
    model: sklearn model, it's the model to save;
    name_model: string, it's the name of the pickle file;
    overwrite: string, whether to overwrite ('yes') or not ('no');
    PATH_DATA: pathlib object, the relative path to save the file to.

    Returns:
    - 0 if executed correctly;
    - 1 if "overwrite" is not 'yes' or 'no'.
    """
    if (overwrite == 'yes'):
        with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
            pkl.dump(model, pkl_file)
    elif (overwrite == 'no'):
        if (not Path(PATH_DATA / name_model).is_file()):
            with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
                pkl.dump(model, pkl_file)
    else:
        raise ValueError("specify whether to overwrite ('yes') or not ('no')")
        return 1

    return 0



def load_model(name_model, PATH_DATA=PATH_DATA):
    """
    Loads a sklearn model

    Parameters
    name_model: string, name of the sklearn model to load;
    PATH_DATA: pathlib object, the relative path to save the file to.

    Returns:
    - sklearn model, if it exists in PATH_DATA;
    - None if the sklearn model doesn't exist.
    """
    try:
        model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
        return model_loaded
    except FileNotFoundError:
        return None



def try_model(X, y, model, name_model, random_state=100, print_score=True, dump_model='yes', PATH_DATA=PATH_DATA):
    """
    Fits a sklearn model to X, y (or loads it if exists in PATH_DATA) and 
    prints the train and test score. It also saves it if "save_model" = 'yes'

    Parameters
    ------------
    X: pandas dataframe, training set;
    y: pandas series, target variable;
    model: sklearn model, to fit on the training set;
    name_model: string, name of the sklearn model;
    random_state: int, random_state of train_test_split;
    score: bool, whether to print the score or not;
    save_model: string, whether to save the model or not;
    PATH_DATA: pathlib object, the relative path to save the file to.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
            test_size=0.25, random_state=random_state)

    model_loaded = load_model(name_model)
    if (model_loaded == None):
        model.fit(X_train, y_train)
        print(model)
        if (dump_model == 'yes'):
            save_model(model, name_model)   
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model.score(X_train, y_train):.4f}\ntest score: {model.score(X_test, y_test):.4f}')

    else:
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model_loaded.score(X_train, y_train):.4f}\ntest score: {model_loaded.score(X_test, y_test):.4f}')












def try_model_old(X, y, model, name_model, random_state=100, print_score=True, save_model='yes', PATH_DATA=PATH_DATA):
    """
    Fits a sklearn model to X, y (or loads it if exists in PATH_DATA) and 
    prints the train and test score. It also saves it if "save_model" = 'yes'

    Parameters
    ------------
    X: pandas dataframe, training set;
    y: pandas series, target variable;
    model: sklearn model, to fit on the training set;
    name_model: string, name of the sklearn model;
    random_state: int, random_state of train_test_split;
    score: bool, whether to print the score or not;
    save_model: string, whether to save the model or not;
    PATH_DATA: pathlib object, the relative path to save the file to.
    """
    # If model is stored in the folder, load it and return the model
    try:
        model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model_loaded.score(X_train, y_train):.4f}\ntest score: {model_loaded.score(X_test, y_test):.4f}')


    # Check if the model was trained on a different PCA decomposition
    except ValueError:
        print(f'ValueError: most likely an existent {name_model} was trained on a different number of components. Delete existing {name_model} and retrain it with the current PCA-decomposed training set')


    # If model not stored in the folder, train it on X_train, y_train
    except FileNotFoundError:
        model.fit(X_train, y_train)
        print(model)
        save_model(model, name_model)

        # if (not Path(PATH_DATA / name_model).is_file()) and (save_model == 'yes'):
        #     with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
        #         pkl.dump(model, pkl_file)
    
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model.score(X_train, y_train):.4f}\ntest score: {model.score(X_test, y_test):.4f}')


