
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


ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters\\regularizers"
MAIN_FOLDER_TRAINING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\pilot_study_may_2021"
MAIN_FOLDER_TESTING_DATA = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\Data\\training_data_february_2021\\interventions"
BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean\\buttons_list")

ROOT_DIR = "/workspace/NeuralNetworkSeptember21/NeuralNetwork/dl-4-tsc-master/tune_parameters/regularizers"
MAIN_FOLDER_TRAINING_DATA = "/workspace/NeuralNetworkSeptember21/Data/pilot_study_may_2021"
MAIN_FOLDER_TESTING_DATA = "/workspace/NeuralNetworkSeptember21/Data/training_data_february_2021/interventions"
BUTTONS_LIST_PATH = os.path.join(MAIN_FOLDER_TRAINING_DATA, "interventions_clean/buttons_list")

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


def write_parameters(path, kernel_dense_l1, kernel_dense_l2, bias_conv, bias_dense, remove_multi_labels):

    parameters_dict = {"kernel_dense_l1": kernel_dense_l1, "kernel_dense_l2": kernel_dense_l2, \
                       "bias_conv": bias_conv, "bias_dense": bias_dense, "remove_multi_labels": remove_multi_labels}

    with open(path, "w") as f:
        json.dump(parameters_dict, f)

if __name__ == "__main__":

   
    parameters_path = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters\\results_2_summary\\parameters_155.json"

    kernel_dense_l1_list = [0, 10e-10, 10e-5]
    kernel_dense_l2_list = [0, 10e-10, 10e-5]
    bias_conv_list = [0, 10e-5, 10e-2, 10e-1]
    bias_dense_list = [0, 10e-5, 10e-2, 10e-1]

    i = 0
    indices = [10, 12, 15, 30, 461]
    indices_where_remove_multi_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521}

    for index in indices:
    
        parameters_path = "/workspace/NeuralNetworkSeptember21/NeuralNetwork/dl-4-tsc-master/tune_parameters/results_2/fcn_multi_labels_{}/parameters.json".format(index)
        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        classifier_name = parameters["classifier_name"]
        feature_combination = parameters["features_columns"]
        stride = parameters["stride"]
        batch_size = parameters["batch_size"]
        dropout_conv1d = parameters["dropout_conv1d"]
        dropout_dense = parameters["dropout_dense"]
        channels_conv1d = parameters["channels_conv1d"]
        time_window_size = parameters["time_window_size"]

        remove_multi_labels = parameters["remove_multi_labels"]

        # Change classifier name
        classifier_name = "fcn_multi_labels_temp"
           
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

        for kernel_dense_l1 in kernel_dense_l1_list :
            for kernel_dense_l2 in kernel_dense_l2_list:
                for bias_conv in bias_conv_list:
                    for bias_dense in bias_dense_list:
                    
                        local_path = str(i)
                        output_directory = os.path.join(ROOT_DIR, local_path)
                        parameters_path = os.path.join(output_directory, "parameters.json")

                        create_directory(output_directory)
                        print("output dir", output_directory)
                        write_parameters(parameters_path, kernel_dense_l1,kernel_dense_l2, bias_conv, bias_dense, remove_multi_labels)

                        create_and_train_classifier(x_train, y_train, x_val, y_val, \
                                output_directory, classifier_name, dropout_conv1d, dropout_dense, \
                                kernel_dense_l1,kernel_dense_l2, bias_conv, bias_dense, \
                                channels_conv1d, batch_size)
                        i += 1



