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




def create_subfile_dr(test_data, pca, model, name_model):

    ids = test_data.index.values + 1
    try:
        test_data_tr = pca.transform(test_data)
    except:
        print('pca transform gone wrong')
        return 1

    try:
        predictions = model.predict(test_data_tr)
    except:
        print('predictions gone wrong')
        return 1

    sub_file = pd.DataFrame({'ImageId': ids, 'Label': predictions})
    sub_file.to_csv(f'submission_{name_model}.csv', sep=',', index=False)

    return 0



def try_model(X, y, model, name_model, PATH_DATA, random_state=100, score=True):

	X_train, X_test, y_train, y_test = train_test_split(X, y,\
			test_size=0.25, random_state=random_state)

	try:
		model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
		if (score):
			print(f'model: {name_model}')
			print(f'train score: {model_loaded.score(X_train, y_train):.4f}\ntest score: {model_loaded.score(X_test, y_test):.4f}')

	except ValueError:
		print(f'ValueError: most likely an existent {name_model} was trained on a different number of components. Delete existing {name_model} and retrain it with the current PCA-decomposed training set')

	except FileNotFoundError:
		model.fit(X_train, y_train)
		if not Path(PATH_DATA / name_model).is_file():
			with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
				pkl.dump(model, pkl_file)
	
		if (score):
			print(f'model: {name_model}')
			print(f'train score: {model.score(X_train, y_train):.4f}\ntest score: {model.score(X_test, y_test):.4f}')



