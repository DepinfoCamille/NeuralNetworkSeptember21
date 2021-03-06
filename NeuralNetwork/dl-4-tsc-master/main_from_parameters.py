
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
from classifiers import fcn_multi_labels, fcn, fcn_multi_labels_temp

import os
import itertools
import json
FIND_FEATURES = True
FIND_NETWORK_PARAMETERS = False


OUTPUT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\results\\fcn_multi_labels"
ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters"
MAIN_FOLDER_TRAINING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\pilot_study_may_2021"
MAIN_FOLDER_TESTING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\training_data_february_2021\\interventions"
BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean\\buttons_list")


ROOT_DIR = "/workspace/NeuralNetworkSeptember21/dl-4-tsc-master/tune_parameters"
OUTPUT_DIR = "/workspace/NeuralNetworkSeptember21/NeuralNetwork/dl-4-tsc-master/results/from_227_79"
#MAIN_FOLDER_TRAINING_DATA = "/workspace/NeuralNetworkSeptember21/Data/pilot_study_may_2021"
#MAIN_FOLDER_TESTING_DATA = "/workspace/NeuralNetworkSeptember21/Data/training_data_february_2021/interventions"
#BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean/buttons_list")

def create_and_train_classifier(x_train, y_train, x_val, y_val, output_directory, classifier_name, \
                                dropout_conv1d, dropout_dense, \
                                kernel_dense_l1, kernel_dense_l2, bias_conv, bias_dense, \
                                channels_conv1d, batch_size):


    test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')
    create_directory(output_directory)

    fit_classifier(x_train, y_train, x_val, y_val, output_directory, \
                   dropout_conv1d, dropout_dense, channels_conv1d, batch_size, \
                   kernel_dense_l1, kernel_dense_l2, bias_conv, bias_dense)

    print('DONE')

def fit_classifier(x_train, y_train, x_val, y_val, output_directory, \
                   dropout_conv1d, dropout_dense, channels_conv1d, batch_size, \
                   kernel_dense_l1, kernel_dense_l2, bias_conv, bias_dense):

    nb_classes = np.unique(np.concatenate((y_train, y_val), axis=0), axis = 0).shape[1]

    # save orignal y because later we will use binary
    y_true = np.argmax(y_val, axis=1)


    input_shape = x_train.shape[1:]

    if classifier_name == 'fcn_multi_labels_temp' and y_train is not None:
        classifier = fcn_multi_labels_temp.Classifier_FCN(output_directory, input_shape, nb_classes, y_train, \
                                                    dropout_conv1D = dropout_conv1d, dropout_dense = dropout_dense, \
                                                    channels_conv1D = channels_conv1d, batch_size = batch_size, 
                                                    kernel_dense_l1 = kernel_dense_l1, kernel_dense_l2 = kernel_dense_l2, \
                                                    bias_conv = bias_conv, bias_dense = bias_dense, \
                                                    verbose = True)

    if classifier_name == 'fcn_multi_labels' and y_train is not None:
        classifier = fcn_multi_labels.Classifier_FCN(output_directory, input_shape, nb_classes, y_train, \
                                                    dropout_conv1D = dropout_conv1d, dropout_dense = dropout_dense, \
                                                    channels_conv1d = channels_conv1d, batch_size = batch_size, \
                                                    verbose = True)

    if classifier_name == 'fcn':
        classifier = fcn.Classifier_FCN(output_directory, input_shape, nb_classes, \
                                        dropout_conv1D = dropout_conv1d, dropout_dense = dropout_dense, \
                                        channels_conv1D = channels_conv1d, batch_size = batch_size, \
                                        verbose = True)

    classifier.fit(x_train, y_train, x_val, y_val, y_true)


def write_parameters(path, classifier_name, features_columns, dropout_conv1d, dropout_dense, channels_conv1d, \
                     batch_size, time_window_size, stride):

    parameters_dict = {"classifier_name": classifier_name, "features_columns": features_columns, \
                       "dropout_conv1d": dropout_conv1d, "dropout_dense": dropout_dense, \
                       "channels_conv1d":channels_conv1d, "batch_size": batch_size, \
                       "time_window_size": time_window_size, "stride": stride}

    with open(path, "w") as f:
        json.dump(parameters_dict, f)

if __name__ == "__main__":

   
    print("helloworld")
    parameters_path = os.path.join(ROOT_DIR, os.path.join("results_2_summary", "parameters_227.json"))
    #"C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters\\results_2_summary\\parameters_227.json"
    regularizers_parameters_path = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters\\regularizers_summary\\regularizers_parameters_79.json"
    regularizers_parameters_path = os.path.join(ROOT_DIR, os.path.join("regularizers_summary", "regularizers_parameters_79.json"))

    with open(parameters_path, "r") as f:
        parameters = json.load(f)
    with open(regularizers_parameters_path, "r") as f:
        regularizers_parameters = json.load(f)

    classifier_name = parameters["classifier_name"]
    feature_combination = parameters["features_columns"]
    stride = parameters["stride"]
    batch_size = parameters["batch_size"]
    dropout_conv1d = parameters["dropout_conv1d"]
    dropout_dense = parameters["dropout_dense"]
    channels_conv1d = parameters["channels_conv1d"]
    time_window_size = parameters["time_window_size"]

    kernel_dense_l1 = regularizers_parameters["kernel_dense_l1"]
    kernel_dense_l2 = regularizers_parameters["kernel_dense_l2"]
    bias_conv = regularizers_parameters["bias_conv"]
    bias_dense = regularizers_parameters["bias_dense"]

    # Change classifier name
    classifier_name = "fcn_multi_labels_temp"
   
    x_train, y_train = compute_sktime_input_from_pilot_study(MAIN_FOLDER_TRAINING_DATA, time_window_size, stride, \
                                                    remove_unannotated_labels = True, 
                                                    features_columns = feature_combination, \
                                                    remove_multi_labels = True, balance_classes=True, \
                                                    gather_classes = None)#classes_to_gather)
    x_val, y_val = compute_sktime_input_from_first_scenarii_data(MAIN_FOLDER_TESTING_DATA, BUTTONS_LIST_PATH, \
                                                            time_window_size, stride, \
                                                            features_columns = feature_combination,
                                                            remove_unannotated_labels = True, \
                                                            gather_classes = None)#classes_to_gather)

    train_cut = x_train.shape[0]//5
    val_cut = x_val.shape[0]//4
    x_val_temp = np.concatenate([x_train[:train_cut, :, :], x_val[:val_cut, :, :]], axis = 0)
    y_val_temp = np.concatenate([y_train[:train_cut, :], y_val[:val_cut, :]], axis = 0)

    x_train = np.concatenate([x_train[train_cut:, :, :], x_val[val_cut:, :, :]], axis = 0)
    y_train = np.concatenate([y_train[train_cut:, :], y_val[val_cut:, :]], axis = 0)

    x_val = x_val_temp
    y_val = y_val_temp

    create_and_train_classifier(x_train, y_train, x_val, y_val, \
                                OUTPUT_DIR, classifier_name, dropout_conv1d, dropout_dense, \
                                kernel_dense_l1,kernel_dense_l2, bias_conv, bias_dense, \
                                channels_conv1d, batch_size)
