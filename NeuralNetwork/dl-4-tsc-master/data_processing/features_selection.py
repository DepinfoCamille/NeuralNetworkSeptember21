import numpy as np
import pandas as pd
import os

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from random import randint
from itertools import combinations

from data_processing import compute_sktime_input_from_first_scenarii_data, compute_sktime_input_from_pilot_study

FEATURES_COLUMNS = ['headLinearVelocityNorm', 'headAngularVelocityNorm', "handIsVisible", 'buttonClicked', \
                    'rightHandVelocityNorm', 'leftHandVelocityNorm', 'gazeDirectionVelocityNorm']

ANNOTATIONS_COLUMNS = ["menu_interaction", "information_assimilation_augmentation", \
                        "information_assimilation_real_world", "real_world_task", "visual_search", \
                        "walk", "compare_real_to_augmentation", "media_manipulation", "position_arrangement"]

def get_random_colors(n):

    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors

def visualize_data(features, labels, n_components = 5):

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(features)
    principalDf = pd.DataFrame(data = principalComponents)#, \
                               #columns = ['principal component 1', 'principal component 2'])

    print("principal1DF")
    print(principalDf)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    unique_labels = np.unique(labels, axis = 0)
    targets = np.unique(labels, axis = 0)
    print("targets", targets)
    colors = get_random_colors(targets.shape[0])
    for target, color in zip(targets,colors):
        indicesToKeep = np.all(labels==target,axis=1)
        print("indices to keep", indicesToKeep)
        ax.scatter(features[indicesToKeep]
                   , finalDf.loc[indicesToKeep, 1]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.show()

def visualize_data(features, labels, n_components = 5):

    pairs_of_features = list(combinations(range(features.shape[1]), 2))
    trio_of_features = list(combinations(range(features.shape[1]), 3))

    #targets = np.unique(labels, axis = 0)
    targets = np.eye(len(ANNOTATIONS_COLUMNS))
    colors = get_random_colors(targets.shape[0])

    for pair in pairs_of_features:

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(FEATURES_COLUMNS[pair[0]], fontsize = 15)
        ax.set_ylabel(FEATURES_COLUMNS[pair[1]], fontsize = 15)
        #ax.set_title('2 component PCA', fontsize = 20)
        for target, color in zip(targets,colors):
            indicesToKeep = np.all(labels==target, axis=1)
            print("indices to keep", indicesToKeep)
            print("features stuff", features[indicesToKeep, pair[1]])
            ax.scatter(features[indicesToKeep, pair[0]]
                       , features[indicesToKeep, pair[1]]
                       , c = color
                       , s = 50
                       , alpha = 0.5)
        ax.legend(ANNOTATIONS_COLUMNS)
        #ax.grid()

        plt.show()


def get_virtual_objects_interactions_by_annotation():
    pass



if __name__ == "__main__":

    # Get data
    main_folder = "C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total"
    time_window_size = 10 #10
    stride = 10 #10

    train_features, train_labels =  compute_sktime_input_from_pilot_study(main_folder, time_window_size, stride, \
                                                           remove_unannotated_labels = False)
    # Takes mean over time
    train_features = np.median(train_features, axis = 2)

    X = train_features
    y = train_labels

    class_balance = np.sum(y, axis = 0)
    print("class balance", class_balance / np.sum(class_balance)*100)


    visualize_data(train_features, train_labels)

    # Select best features
    selector = SelectKBest(f_classif, k=4)
    selected_features = selector.fit_transform(train_features, train_labels)
    print("selected_features", selected_features)
    f_score_indexes = (-selector.scores_).argsort()[:10]

    # fit random forest model
    model = RandomForestRegressor(n_estimators=500, random_state=1)
    model.fit(X, y)
    # show importance scores
    print(model.feature_importances_)
    # plot importance scores
    #names = dataframe.columns.values[0:-1]
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, model.feature_importances_)
    #pyplot.xticks(ticks, names)
    plt.show()

    # Check if they are the same on the testing set