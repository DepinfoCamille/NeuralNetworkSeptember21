import numpy as np
import pandas as pd

import os
import glob
import json
from collections import defaultdict

LINUX = True

ANNOTATIONS_COLUMNS = ["menu_interaction", "information_assimilation_augmentation", \
                        "information_assimilation_real_world", "real_world_task", "visual_search", \
                        "walk", "compare_real_to_augmentation", "media_manipulation", "position_arrangement"]

ANNOTATIONS_COLUMNS2 = ["interaction_with_virtual", "assimilation_virtual", "real_world_actions", "displacement", \
                       "visual_search"]

ANNOTATIONS_COLUMNS2 = ANNOTATIONS_COLUMNS

FEATURES_COLUMNS = ['headLinearVelocityNorm', 'headAngularVelocityNorm', "handIsVisible", 'buttonClicked', \
                    'rightHandVelocityNorm', 'leftHandVelocityNorm', 'gazeDirectionVelocityNorm']


def load_ids(interventions_folder, annotations_folder = ""):
    ids = []
    regex = os.path.join(interventions_folder, "data_*.csv")
    offset = len("data_")
    for path in glob.glob(regex):
        if LINUX:
            id = int(path.split("/")[-1][offset:offset+4])
        else:
            id = int(path.split("\\")[-1][offset:offset+4])
        if len(annotations_folder):
           if os.path.exists(os.path.join(annotations_folder, "annotation_{}.json".format(id))):
            ids.append(id)
        else:
            ids.append(id)

    return ids
            
def load_data(interventions_folder, intervention_id):
    data_path = os.path.join(interventions_folder, "intervention_clean_" + str(intervention_id) + ".json")
    if not os.path.exists(data_path):
        data_path = os.path.join(interventions_folder, "intervention_" + str(intervention_id) + ".json")

    with open(data_path, "r") as f:
        data = json.load(f)


    # Load timestamps
    # When gap between two timestamps is too big, it probably means the video was being started
    # We are not interested for tracked data before the video is started 
    start_index = 0
    if "timestamps" in data.keys():
        data["timestamp"] = data.pop("timestamps")
    for i in range(1, len(data["timestamp"])):
        if data["timestamp"][i] - data["timestamp"][i-1] > 10.:
            print("pb", i, data["timestamp"][i+2], data["timestamp"][i+1], data["timestamp"][i], data["timestamp"][i-1])
            start_index = i
    
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = np.array(v[start_index:])

    return data

def load_buttons_mapping(path):

    # Load buttons name
    with open(path, "r") as file:
        content = set(file.readlines())
    buttons_list = [x[:-1] for x in content if len(x) > 1] # remove \n
    # None in first position
    buttons_list.remove("None")
    buttons_list.insert(0, "None")
    
    return buttons_list

def load_annotation_data(annotations_folder, intervention_id, annotations_columns = []):

    annotation_path = os.path.join(annotations_folder, "annotation_{}.json".format(intervention_id))
    annotation_data_dict = defaultdict()
    annotation_data = pd.DataFrame()

    with open(annotation_path, "r") as f:
        annotation_data_dict = json.load(f)

    annotation_data_dict.pop("end_timestamp_index")
    start_index = annotation_data_dict.pop("start_timestamp_index")
    end_index = annotation_data_dict.pop("end_video_index")

    start_index = start_index if start_index > 0 else 0
    end_index = end_index if start_index > 0 else len(annotation_data_dict["menu_interaction"])

    for k, v in annotation_data_dict.items():
        if isinstance(v, list):
            annotation_data_dict[k] = np.array(v[start_index:end_index])

    annotation_data = annotation_data.from_dict(annotation_data_dict, orient='columns')

    for annotation_col in ANNOTATIONS_COLUMNS:
        if annotation_col not in annotation_data.columns:
            annotation_data[annotation_col] = 0

    return annotation_data if not len(ANNOTATIONS_COLUMNS) else annotation_data[ANNOTATIONS_COLUMNS]

def load_intervention_data(interventions_folder, intervention_id, invalid_step_ids =  [], selected_features = []):

    features_path = os.path.join(interventions_folder, "data_" + str(intervention_id) + ".csv")
    intervention_data = defaultdict()
    features_data = pd.DataFrame()

    # Load data
    intervention_data = load_data(interventions_folder, intervention_id)
    features_data = pd.read_csv(features_path)


    if len(invalid_step_ids):
        # Get timestamps when step changes
        step_change_timestamps = intervention_data["timestamp"][np.nonzero(intervention_data["stepChange"])[0]]
        #print("remove data")
        #print(features_data.shape)
        # Only keep valid parts
        for i, invalid_step_id in enumerate(invalid_step_ids):
            lower_bound = step_change_timestamps[invalid_step_id]
            upper_bound = step_change_timestamps[invalid_step_id + 1] if  i < len(step_change_timestamps)-1 else \
                          intervention_data["timestamp"][-1]
            #features_data = features_data[(features_data["timestamp"] < lower_bound) | (features_data["timestamp"] > upper_bound)]
        print(features_data.shape)

    return features_data if not len(selected_features) else features_data[selected_features]


def intervention_data_as_feature_vectors(intervention_data, buttons_list):

    mapping = {k: v for v, k in enumerate(buttons_list)}

    for column_name in ["buttonClicked", "hitByHandPointer", "hitByGazePointer"]:
        if column_name in intervention_data.columns:
            intervention_data[column_name] = intervention_data[column_name].map(mapping)
            intervention_data[column_name] = intervention_data[column_name].astype(float).fillna(0.)

    intervention_data["handIsVisible"] = intervention_data.apply(lambda x: float(x["rightHandPositionNorm"] > 0) \
                                                                        + float(x["leftHandPositionNorm"] > 0), axis = 1)
    return intervention_data


def gather_labels(data, gathering_dict):
    for k, v in gathering_dict.items():
        data[k] = data.apply(lambda x: 1 if x[v].sum() else 0, axis = 1)# data[v].sum(axis = 1)
        data.drop(v, axis = 1, inplace = True)

    return data


def concatenate_interventions(interventions_folder, annotations_folder = "", \
                              invalid_step_ids = None):

    ids = []
    interventions_data = []
    has_annotations_in_folder = len(annotations_folder)# and len(ANNOTATIONS_COLUMNS)

    #Load ids
    ids = load_ids(interventions_folder, annotations_folder)

    # Load interventions and annotations
    for id in ids:

        if invalid_step_ids is not None and str(id) in invalid_step_ids:
            intervention_data = load_intervention_data(interventions_folder, id , \
                                                        invalid_step_ids = invalid_step_ids[str(id)])
        else:
            intervention_data = load_intervention_data(interventions_folder, id)
        #intervention_data = load_intervention_data(interventions_folder, id)
        if has_annotations_in_folder:
            annotation_data = load_annotation_data(annotations_folder, id)
            intervention_data = pd.concat([intervention_data, annotation_data], axis = 1)
            intervention_data.dropna(inplace = True)
        intervention_data["interventionId"] = id
        interventions_data.append(intervention_data)

    return pd.concat(interventions_data, axis = 0)

def from_single_to_multiple_annotation_columns(data,  annotations_columns = []):

    if "annotation" not in data.columns:
        for col in ANNOTATIONS_COLUMNS:
            if col not in data.columns:
                data[col] = 0
        return data
    
    for col in ANNOTATIONS_COLUMNS:
        data[col] = data.apply(lambda x: 0 if x["annotation"] != col else 1, axis = 1)

    return data.drop("annotation", axis = 1)


# If stride < time_window_size, time series overlap each other
def data_as_sktime_input(features, time_window_size, stride, features_columns = [], annotations_columns = [],\
                         remove_unannotated_labels = False):

    X = []
    y = []

    for intervention_id, data in features.groupby("interventionId", sort = False):

        # Remove na values
        data.dropna(subset = features_columns, inplace = True)

        X_i = data.dropna().copy().select_dtypes(float) if not len(features_columns) else \
              data[features_columns].copy().select_dtypes(float)

        if X_i is not None and X_i.shape[0] >=  time_window_size:

            # Order columns and convert to numpy
            X_i = X_i.to_numpy()
            y_i = data[ANNOTATIONS_COLUMNS2].to_numpy()

            # Remove unnanotated classes
            wild = (y_i == 0).all(axis=1)
            if remove_unannotated_labels:
                y_i = np.delete(y_i, wild, axis = 0)
                X_i = np.delete(X_i, wild, axis = 0)

            else:
                # Change zeros rows to ones rows
                y_i[wild, :] = np.ones(( y_i.shape[1]))

            if X_i.shape[0] >=  time_window_size:
                temp_X = [X_i[i: i+time_window_size, :] for i in range(0, X_i.shape[0] - time_window_size, stride)]
                temp_y = [y_i[i+time_window_size] for i in range(0, X_i.shape[0] - time_window_size, stride)]
                #print("temp_y 0", temp_y)
                #print("temp_X 0", temp_X)


                indices = [i for i in range(0, X_i.shape[0] - time_window_size, stride) \
                             if np.sum(np.abs(y_i[i+time_window_size, :] - np.mean(y_i[i:i+time_window_size, :], axis = 0))) < 0.1]
                #print("indices", indices)
                temp_X = [X_i[i: i+time_window_size, :] for i in indices]
                temp_y = [y_i[i+time_window_size] for i in indices]
                #print("temp_X 1", temp_X)
                #print("temp_y 1", temp_y)


                #for i in range(0, X_i.shape[0] - time_window_size, stride):
                #    # We add the sample only if there is no class change within corresponding window size
                #    mean_over_window =  np.mean(y_i[i:i+time_window_size, :], axis = 0)
                #    if np.sum(np.abs(y_i[i+time_window_size, :] - mean_over_window)) < 0.1:
                ##        temp_X.append(X_i[i: i+time_window_size, :])
                ##        temp_y.append(y_i[i+time_window_size, :])
                #        print("yo", mean_over_window, temp_y[-1], y_i[i+time_window_size, :] - mean_over_window, np.sum(y_i[i+time_window_size, :] - mean_over_window))

                if len(temp_y):
                    X.append(np.swapaxes(np.stack(temp_X, axis = 0) , 1, 2))
                    y.append(np.stack(temp_y, axis = 0))

    X = np.concatenate(X, axis = 0)
    y = np.concatenate(y, axis = 0)     

    return X, y

def compute_sktime_input_from_pilot_study(main_folder, time_window_size = 10, stride = 10, \
                                          features_columns = [], annotations_columns = [], gather_classes = None, \
                                          remove_unannotated_labels = False, \
                                          remove_multi_labels = False, balance_classes = False):

    print("main_folder", main_folder)
    interventions_folder = os.path.join(main_folder, "interventions_clean") #"C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total
    annotations_folder =  os.path.join(main_folder, "annotations")
    buttons_list_path = os.path.join(interventions_folder, "buttons_list")
    unwanted_steps_path = os.path.join(annotations_folder, "unwanted_steps_for_training.json")

    with open(unwanted_steps_path, "r") as f:
        unwanted_steps_dict = json.load(f)

    buttons_list = load_buttons_mapping(buttons_list_path)
    data = concatenate_interventions(interventions_folder, annotations_folder,\
                                     invalid_step_ids = unwanted_steps_dict)

    return compute_sktime_input_2(data, buttons_list, time_window_size, stride, \
                                  remove_unannotated_labels = remove_unannotated_labels, \
                                  features_columns = features_columns, annotations_columns = annotations_columns, \
                                  gather_classes = gather_classes,\
                                  remove_multi_labels = remove_multi_labels, balance_classes = balance_classes)


def compute_sktime_input_from_first_scenarii_data(interventions_folder, buttons_list_path, time_window_size = 10, \
                                                  stride = 10,  features_columns = [], annotations_columns = [], \
                                                  gather_classes = None, \
                                                  remove_unannotated_labels = False, remove_multi_labels = False, \
                                                  balance_classes = False):

    data = concatenate_interventions(interventions_folder)
    buttons_list = load_buttons_mapping(buttons_list_path)

    return compute_sktime_input_2(data, buttons_list, time_window_size, stride, \
                                  remove_unannotated_labels = remove_unannotated_labels, \
                                  features_columns = features_columns, annotations_columns = annotations_columns, gather_classes = gather_classes,
                                  remove_multi_labels = remove_multi_labels, balance_classes = balance_classes)

def compute_sktime_input_2(data, buttons_list, time_window_size = 10, stride = 10, \
                           features_columns = [], annotations_columns = [], gather_classes = None,\
                           remove_unannotated_labels = False, remove_multi_labels = False, balance_classes = False):


    data = from_single_to_multiple_annotation_columns(data)
    # na values to change into 0 if some annotations columns exist for some interventions but not for others
    data[ANNOTATIONS_COLUMNS] = data[ANNOTATIONS_COLUMNS].fillna(0)

    if gather_classes is not None:
        data = gather_labels(data, gather_classes)


    feature_vectors = intervention_data_as_feature_vectors(data, buttons_list)
    X, y = data_as_sktime_input(data, time_window_size, stride, features_columns = features_columns, \
                                remove_unannotated_labels = remove_unannotated_labels)

    if remove_multi_labels:
        print("X shape before multi labels", X.shape)
        indices_to_remove = np.where(np.sum(y, axis = 1) > 1)
        X = np.delete(X, indices_to_remove, axis = 0)
        y = np.delete(y, indices_to_remove, axis = 0)

        print("X shape after multi labels", X.shape)


    if balance_classes:

        print("X before balance", X.shape)

        nb_samples_per_class = np.sum(y, axis = 0)
        class_weights = nb_samples_per_class / np.sum(nb_samples_per_class)
        print("class_weights before balance", class_weights)
        print("nb_samples_per_class before balance", nb_samples_per_class)
        to_reduce = []
        thresh = 2. / len(class_weights)
        for i in range(len(class_weights)):
            if class_weights[i] >= thresh:
                to_reduce.append(i)

        # We remove some of the samples of classesd too represented
        for i in to_reduce:
            indices = np.where((y[:, i] > 0) & (np.sum(y, axis = 1) < 2))
            # Shuffle the indices, select the first 30% (rounded down with int())
            print("nb_samples_per_class[i] - thresh", nb_samples_per_class[i] - thresh*np.sum(nb_samples_per_class)/2)
            to_replace = np.random.permutation(indices)[0, :max(nb_samples_per_class[i] - int(thresh*np.sum(nb_samples_per_class)/2), nb_samples_per_class[i]//2)]
            X = np.delete(X, to_replace, axis = 0)
            y = np.delete(y, to_replace, axis = 0)

        print("X after balance", X.shape)

        nb_samples_per_class = np.sum(y, axis = 0)
        class_weights = nb_samples_per_class / np.sum(nb_samples_per_class)
        print("class_weights after balance", class_weights)
        print("nb_samples_per_class after balance", nb_samples_per_class)


    return X, y


if __name__ == "__main__":

    interventions_folder = "C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total\\interventions_clean"
    annotations_folder = "C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total\\annotations"
    buttons_list_path = os.path.join(interventions_folder, "buttons_list")

    time_window_size = 10 
    stride = time_window_size

    data = concatenate_interventions(interventions_folder, annotations_folder)
    buttons_list = load_buttons_mapping(buttons_list_path)
    feature_vectors = intervention_data_as_feature_vectors(data, buttons_list)
    X, y = data_as_sktime_input(data, time_window_size, stride, FEATURES_COLUMNS)

    print(feature_vectors)
    print(X)
    print(y)

    training_data =  compute_sktime_input_from_pilot_study( "C:\\Users\\Camille\\Documents\\These\\PilotStudyMay21\\results_total", \
                                                            time_window_size, stride, \
                                                            remove_unannotated_labels = False, remove_multi_labels = True, \
                                                            balance_classes=True)
