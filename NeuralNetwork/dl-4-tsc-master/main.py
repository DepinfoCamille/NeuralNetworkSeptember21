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


def fit_classifier(training_set, testing_set):
    x_train = training_set[0]
    y_train = training_set[1]
    x_test = testing_set[0]
    y_test = testing_set[1]

    # Shuffle samples
    p = np.random.permutation(x_train.shape[0])
    x_train = x_train[p, :, :]
    y_train = y_train[p, :]

    #start = x_train.shape[0]//4
    #x_test = x_train[:start, :, :]
    #y_test = y_train[:start, :]

    #x_train = x_train[start:, :, :]
    #y_train = y_train[start, :]

    cut = x_test.shape[0]//3
    x_train = np.concatenate([x_train, x_test[:cut, :, :]], axis = 0)
    y_train = np.concatenate([y_train, y_test[:cut, :]], axis = 0)

    x_test = x_test[cut:, :, :]
    y_test = y_test[cut:, :]

    print("x_train", x_train.shape)
    print("y_train", y_train.shape)

    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    nb_classes = np.unique(np.concatenate((y_train, y_test), axis=0), axis = 0).shape[1]

    # transform the labels from integers to one hot vectors
    #enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    #enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    #y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    #y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    print("input_shape", input_shape, "nb_classes", nb_classes)
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = True, y_train = y_train)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, y_train = None):

    if classifier_name == 'fcn_multi_labels' and y_train is not None:
        from classifiers import fcn_multi_labels
        return fcn_multi_labels.Classifier_FCN(output_directory, input_shape, nb_classes, y_train, verbose)

    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master"
main_folder_testing_data = "C:\\Users\\Camille\\Documents\\These\\Data\\Test_tout"
main_folder_training_data = "C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total"
buttons_list_path = os.path.join(main_folder_training_data, "interventions\\buttons_list")
time_window_size = 5 #15
stride = 5 # 15

features_columns = ['headLinearVelocityNorm', 'headAngularVelocityNorm', "handIsVisible", 'buttonClicked', \
                    'rightHandVelocityNorm', 'leftHandVelocityNorm', 'gazeDirectionVelocityNorm']



what_to_do = "fit_classifier" #"run_all"#

assert what_to_do in {"run_all", 'transform_mts_to_ucr_format', 'visualize_filter', \
                      'viz_for_survey_paper', 'viz_cam', 'generate_results_csv', "fit_classifier"}

if what_to_do == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)
            print("dataset_dict")

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)

                    fit_classifier()

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

elif what_to_do == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif what_to_do == 'visualize_filter':
    visualize_filter(root_dir)
elif what_to_do == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif what_to_do == 'viz_cam':
    viz_cam(root_dir)
elif what_to_do == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
elif what_to_do == 'fit_classifier':
    # this is the code used to launch an experiment on a dataset
    archive_name = what_to_do
    classifier_name = "fcn"
    classifier_name = "fcn_multi_labels"

    output_directory = os.path.join(root_dir, "results\\{}".format(classifier_name))
    test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')

    if False:# os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        classes_to_gather = \
        {
        "interaction_with_virtual": ["menu_interaction", "media_manipulation"],
        "assimilation_virtual": ["information_assimilation_augmentation", "compare_real_to_augmentation"],
        "real_world_actions": ["information_assimilation_real_world", "real_world_task"],
        "displacement":["walk", "position_arrangement"]
        }
        training_data =  compute_sktime_input_from_pilot_study(main_folder_training_data, time_window_size, stride, \
                                                               remove_unannotated_labels = True, 
                                                               features_columns = features_columns, \
                                                                remove_multi_labels = True, balance_classes=True, \
                                                                gather_classes = None)#classes_to_gather)
        testing_data = compute_sktime_input_from_first_scenarii_data(main_folder_testing_data, buttons_list_path, \
                                                                     time_window_size, stride, \
                                                                     features_columns = features_columns,
                                                                     remove_unannotated_labels = True, \
                                                                     gather_classes = None)#classes_to_gather)

        fit_classifier(training_data, testing_data)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
