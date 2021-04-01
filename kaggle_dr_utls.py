from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, \
BaggingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.decomposition import TruncatedSVD, PCA

def plot_img(data, nrows=3, ncols=3, rows=28, cols=28):

    np.random.seed(10)

    fig, ax = plt.subplots(ncols, nrows, figsize=(ncols*nrows, ncols*nrows))
    ax = ax.ravel()

    rnd_idx = np.random.randint(0, len(data), size=nrows*ncols)

    data_reshaped = np.array(data).reshape(-1, 28, 28, 1)
    for i, idx in enumerate(rnd_idx):
        ax[i].imshow(data_reshaped[idx])

    return fig, ax




def create_subfile_dr(test_data, pca, model):

    ids = test_data.index.values + 1
    test_data_tr = pca.transform(test_data)

    predictions = model.predict(test_data_tr)

    sub_file = pd.DataFrame({'ImageId': ids, 'Label': predictions})
    sub_file.to_csv(f'submission_{model}.csv', sep=',', index=False)

    return 0
