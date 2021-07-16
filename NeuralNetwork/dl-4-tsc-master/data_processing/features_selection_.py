from shutil import copyfile, copytree
from glob import glob
from os.path import join
import pandas as pd


ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters"
TARGET_DIR = join(ROOT_DIR, "results_2_summary")

FINE_TUNE = False

if __name__ == "__main__":


    all_results = pd.DataFrame()
    if FINE_TUNE:
        for id in [2, 9, 10, 29, 75, 97, 102, 105, 108, 109, \
                   110, 114, 123, 132, 137, 159, 186, 213, 231, 240, 249]:
            values = pd.read_csv(join(ROOT_DIR, "results_2\\fcn_multi_labels_{}\\history.csv").format(id))
            values["id"] = id
            best_row = values.loc[(values["f1_score"] + 2*values["val_f1_score"]).argmax(axis = 0), :]
            #print("best_row", best_row.columns)
            all_results = all_results.append(best_row)# pd.concat([all_results, best_row], axis = 0)

        all_results.sort_values(["val_f1_score", "f1_score"], axis=0, inplace = True)

        all_results.to_csv(join(TARGET_DIR, "summary.csv"))
            

    else:

        offset = len("fcn_multi_labels_")
        for file_path in glob(join(ROOT_DIR, "results_2\\fcn_*")):
            id = int(file_path.split("\\")[-1][offset:])
            source_epochs_loss = join(file_path, "epochs_loss.png")
            source_parameters = join(file_path, "parameters.json")
            target_epochs_loss = join(TARGET_DIR, "epochs_loss_{}.png".format(id))
            target_parameters = join(TARGET_DIR, "parameters_{}.json".format(id))

            copyfile(source_epochs_loss, target_epochs_loss)        
            copyfile(source_parameters, target_parameters)


       