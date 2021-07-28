
from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
from data_processing.data_processing import compute_sktime_input_from_first_scenarii_data, compute_sktime_input_from_pilot_study
from classifiers import fcn_multi_labels, fcn

import os
import itertools
import json
FIND_FEATURES = True
FIND_NETWORK_PARAMETERS = False


ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters"
MAIN_FOLDER_TRAINING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\pilot_study_may_2021"
MAIN_FOLDER_TESTING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\training_data_february_2021\\interventions"
BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean\\buttons_list")

ROOT_DIR = "/workspace/NeuralNetworkSeptember21/NeuralNetwork/dl-4-tsc-master/tune_parameters"
MAIN_FOLDER_TRAINING_DATA = "/workspace/NeuralNetworkSeptember21/Data/pilot_study_may_2021"
MAIN_FOLDER_TESTING_DATA = "/workspace/NeuralNetworkSeptember21/Data/training_data_february_2021/interventions"
BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean/buttons_list")

def create_and_train_classifier(x_train, y_train, x_val, y_val, output_directory, classifier_name, \
                                dropout_conv1d, dropout_dense, channels_conv1d, batch_size):


    test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')
    create_directory(output_directory)

    fit_classifier(x_train, y_train, x_val, y_val, dropout_conv1D, dropout_dense, channels_conv1d, batch_size)

    print('DONE')

def fit_classifier(x_train, y_train, x_val, y_val, dropout_conv1D, dropout_dense, channels_conv1d, batch_size):

    nb_classes = np.unique(np.concatenate((y_train, y_val), axis=0), axis = 0).shape[1]

    # save orignal y because later we will use binary
    y_true = np.argmax(y_val, axis=1)


    input_shape = x_train.shape[1:]
    if classifier_name == 'fcn_multi_labels' and y_train is not None:
        classifier = fcn_multi_labels.Classifier_FCN(output_directory, input_shape, nb_classes, np.concatenate([y_train, y_val], axis = 0), \
                                                    dropout_conv1D = dropout_conv1D, dropout_dense = dropout_dense, \
                                                    channels_conv1d = channels_conv1d, batch_size = batch_size, \
                                                    verbose = True)

    if classifier_name == 'fcn':
        classifier = fcn.Classifier_FCN(output_directory, input_shape, nb_classes, \
                                        dropout_conv1D = dropout_conv1D, dropout_dense = dropout_dense, \
                                        channels_conv1d = channels_conv1d, batch_size = batch_size, \
                                        verbose = True)

    classifier.fit(x_train, y_train, x_val, y_val, y_true)


def write_parameters(path, classifier_name, features_columns, dropout_conv1d, dropout_dense, channels_conv1d, \
                     batch_size, time_window_size, stride, remove_multi_labels):

    parameters_dict = {"classifier_name": classifier_name, "features_columns": features_columns, \
                       "dropout_conv1d": dropout_conv1d, "dropout_dense": dropout_dense, \
                       "channels_conv1d":channels_conv1d, "batch_size": batch_size, \
                       "time_window_size": time_window_size, "stride": stride, "remove_multi_labels": remove_multi_labels}

    with open(path, "w") as f:
        json.dump(parameters_dict, f)

if __name__ == "__main__":

    all_features =  ['headLinearVelocityNorm', 'headAngularVelocityNorm', "handIsVisible", 'buttonClicked', \
                     'rightHandVelocityNorm', 'leftHandVelocityNorm', 'gazeDirectionVelocityNorm']

    features_combinations = []
    for i in range(5, 7):
        for comb in itertools.combinations(all_features, i):
            features_combinations.append(list(comb))

    classifier_names = ["fcn_multi_labels", "fcn"]

    gather_classes = [False, True]
    conv1D_channels = [64, 128, 256]
    dropout_conv1D = [0., 0.1, 0.5]
    dropout_dense = [0., 0.5, 0.8]
    batch_sizes = [4, 8, 16]
    time_windows_and_strides = [(5, 5), (10, 10), (15, 20)]



    classes_to_gather = \
    {
    "interaction_with_virtual": ["menu_interaction", "media_manipulation"],
    "assimilation_virtual": ["information_assimilation_augmentation", "compare_real_to_augmentation"],
    "real_world_actions": ["information_assimilation_real_world", "real_world_task"],
    "displacement":["walk", "position_arrangement"]
    }

    # Default parameters, can be overwritten 
    classifier_name = "fcn_multi_labels"
    dropout_conv1d = 0.2
    dropout_dense = 0.6
    channels_conv1d = 128
    time_window_size = 10
    stride = 5 # à partie de 43, sinon c'était 10 (car j'ai viré les border data)
    batch_size = 16

    #i = 0
    ## Find optimal combination of features with default neural network parameters
    #for feature_combination in features_combinations:
    #    for time_window_size, stride in time_windows_and_strides:
    #        if i >= len(batch_sizes) * 120//len(batch_sizes):
    #            print("YO", i)
    #            x_train, y_train = compute_sktime_input_from_pilot_study(MAIN_FOLDER_TRAINING_DATA, time_window_size, stride, \
    #                                                            remove_unannotated_labels = True, 
    #                                                            features_columns = feature_combination, \
    #                                                            remove_multi_labels = True, balance_classes=True, \
    #                                                            gather_classes = None)#classes_to_gather)
    #            x_val, y_val = compute_sktime_input_from_first_scenarii_data(MAIN_FOLDER_TESTING_DATA, BUTTONS_LIST_PATH, \
    #                                                                    time_window_size, stride, \
    #                                                                    features_columns = feature_combination,
    #                                                                    remove_unannotated_labels = True, \
    #                                                                    gather_classes = None)#classes_to_gather)

    #            train_cut = x_train.shape[0]//5
    #            val_cut = x_val.shape[0]//4
    #            x_val_temp = np.concatenate([x_train[:train_cut, :, :], x_val[:val_cut, :, :]], axis = 0)
    #            y_val_temp = np.concatenate([y_train[:train_cut, :], y_val[:val_cut, :]], axis = 0)

    #            x_train = np.concatenate([x_train[train_cut:, :, :], x_val[val_cut:, :, :]], axis = 0)
    #            y_train = np.concatenate([y_train[train_cut:, :], y_val[val_cut:, :]], axis = 0)

    #            x_val = x_val_temp
    #            y_val = y_val_temp


    #        for batch_size in batch_sizes:

    #            if i > 120:

    #                print("you", i)

    #                output_directory = os.path.join(ROOT_DIR, "results\\{}_{}".format(classifier_name, i))
    #                parameters_path = os.path.join(output_directory, "parameters.json")

    #                create_directory(output_directory)

    #                write_parameters(parameters_path, classifier_name, feature_combination, dropout_conv1d, dropout_dense, \
    #                                 channels_conv1d, batch_size, time_window_size, stride)
    #                create_and_train_classifier(x_train, y_train, x_val, y_val, \
    #                                            output_directory, classifier_name, dropout_conv1d, dropout_dense, \
    #                                            channels_conv1d, batch_size)

    #            i += 1


    # Second bath of parameters tuning

    already_fine_tuned = \
    {
    "97": {"classifier_name": "fcn_multi_labels", "features_columns": ["headLinearVelocityNorm", "handIsVisible", "buttonClicked", "rightHandVelocityNorm", "leftHandVelocityNorm"], "dropout_conv1d": 0.2, "dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 8, "time_window_size": 15, "stride": 20},
    "105": {"classifier_name": "fcn_multi_labels", "features_columns": ["headLinearVelocityNorm", "handIsVisible", "buttonClicked", "rightHandVelocityNorm", "gazeDirectionVelocityNorm"], "dropout_conv1d": 0.2, "dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 4, "time_window_size": 15, "stride": 20}, 
    "114": {"classifier_name": "fcn_multi_labels", "features_columns": ["headLinearVelocityNorm", "handIsVisible", "buttonClicked", "leftHandVelocityNorm", "gazeDirectionVelocityNorm"], "dropout_conv1d": 0.2, "dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 4, "time_window_size": 15, "stride": 20},
    "123": {"classifier_name": "fcn_multi_labels", "features_columns": ["headLinearVelocityNorm", "handIsVisible", "rightHandVelocityNorm", "leftHandVelocityNorm", "gazeDirectionVelocityNorm"], "dropout_conv1d": 0.2, "dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 4, "time_window_size": 15, "stride": 20},
    "159": {"classifier_name": "fcn_multi_labels", "features_columns": [ "headAngularVelocityNorm", "handIsVisible", "buttonClicked", "leftHandVelocityNorm", "gazeDirectionVelocityNorm" ],"dropout_conv1d": 0.2,"dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 4, "time_window_size": 15, "stride": 20},
    "231": {"classifier_name": "fcn_multi_labels", "features_columns": ["headLinearVelocityNorm", "headAngularVelocityNorm", "buttonClicked", "rightHandVelocityNorm", "leftHandVelocityNorm", "gazeDirectionVelocityNorm"], "dropout_conv1d": 0.2, "dropout_dense": 0.6, "channels_conv1d": 128, "batch_size": 4, "time_window_size": 15, "stride": 20}
    }


    parameters_to_test = [["headLinearVelocityNorm", "handIsVisible", "buttonClicked", "handVelocityNorm"], \
                          ["headLinearVelocityNorm", "handIsVisible", "buttonClicked", "handVelocityNorm", "gazeDirectionVelocityNorm"], \
                          ["headLinearVelocityNorm", "handIsVisible", "handVelocityNorm", "gazeDirectionVelocityNorm"], \
                          ["headLinearVelocityNorm", "headAngularVelocityNorm", "buttonClicked", "handVelocityNorm", "gazeDirectionVelocityNorm"], 
                          ["headLinearVelocityNorm", "headAngularVelocityNorm", "handIsVisible", "buttonClicked", "handVelocityNorm", "gazeDirectionVelocityNorm"]]

    # Default parameters, can be overwritten 
    dropout_conv1D_list = [0.1, 0.5]
    dropout_dense_list = [0.3, 0.5, 0.8]
    channels_conv1d_list = [64, 128, 256]
    strides = [2, 4, 6]
    remove_multi_labels_list = [True, False]
    i = 0
    x_train = None
    # Find optimal combination of features with default neural network parameters
    #for parameters in already_fine_tuned.values():
    for feature_combination in parameters_to_test:

        #time_window_size = parameters["time_window_size"]
        #stride = parameters["stride"]
        #feature_combination = parameters["features_columns"]
        #batch_size = parameters["batch_size"]
        time_window_size = 15
        batch_size = 4

        for stride in strides:
            for remove_multi_labels in remove_multi_labels_list:
                for dropout_conv1D in dropout_conv1D_list:
                    for dropout_dense in dropout_dense_list:
                        for channels_conv1d in channels_conv1d_list:

                        #for remove_multi_labels in remove_multi_labels_list:

                            if x_train is None:

                                x_train_0, y_train_0 = compute_sktime_input_from_pilot_study(MAIN_FOLDER_TRAINING_DATA + "_train", time_window_size, stride, \
                                                                                remove_unannotated_labels = True, 
                                                                                features_columns = feature_combination, \
                                                                                remove_multi_labels = remove_multi_labels, balance_classes=True, \
                                                                                gather_classes = None)#classes_to_gather)

                                x_train_1, y_train_1 = compute_sktime_input_from_first_scenarii_data(MAIN_FOLDER_TESTING_DATA + "_train", BUTTONS_LIST_PATH, \
                                                                                        time_window_size, stride, \
                                                                                        features_columns = feature_combination, \
                                                                                        remove_multi_labels = remove_multi_labels,
                                                                                        remove_unannotated_labels = True, \
                                                                                        gather_classes = None)#classes_to_gather)

                                x_val_0, y_val_0 = compute_sktime_input_from_pilot_study(MAIN_FOLDER_TRAINING_DATA  + "_val", time_window_size, stride, \
                                                                                remove_unannotated_labels = True,
                                                                                features_columns = feature_combination, \
                                                                                remove_multi_labels = remove_multi_labels, balance_classes=True, \
                                                                                gather_classes = None)#classes_to_gather)

                                x_val_1, y_val_1 = compute_sktime_input_from_first_scenarii_data(MAIN_FOLDER_TESTING_DATA  + "_val", BUTTONS_LIST_PATH, \
                                                                                        time_window_size, stride, \
                                                                                        features_columns = feature_combination,
                                                                                        remove_multi_labels = remove_multi_labels, 
                                                                                        remove_unannotated_labels = True, \
                                                                                        gather_classes = None)#classes_to_gather)

                                x_train = np.concatenate([x_train_0, x_train_1], axis = 0)
                                y_train = np.concatenate([y_train_0, y_train_1], axis = 0)

                                x_val = np.concatenate([x_val_0, x_val_1], axis = 0)
                                y_val = np.concatenate([y_val_0, y_val_1], axis = 0)
                                print("x_train", x_train.shape)
                                print("x_val", x_val.shape)


                                local_path = os.path.join("results_2", "{}_{}".format(classifier_name, i))
                                output_directory = os.path.join(ROOT_DIR, local_path)
                                parameters_path = os.path.join(output_directory, "parameters.json")

                                create_directory(output_directory)
                                print("output dir", output_directory)
                                write_parameters(parameters_path, classifier_name, feature_combination, dropout_conv1d, dropout_dense, \
                                                    channels_conv1d, batch_size, time_window_size, stride, remove_multi_labels)
                                create_and_train_classifier(x_train, y_train, x_val, y_val, \
                                                            output_directory, classifier_name, dropout_conv1d, dropout_dense, \
                                                            channels_conv1d, batch_size)

                            i += 1
                x_train = None                                     
