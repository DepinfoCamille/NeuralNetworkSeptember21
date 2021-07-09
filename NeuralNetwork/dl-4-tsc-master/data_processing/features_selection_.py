from shutil import copyfile
from glob import glob
from os.path import join

ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters"
TARGET_DIR = join(ROOT_DIR, "results_summary")

if __name__ == "__main__":


    offset = len("fcn_multi_labels_")
    for file_path in glob(join(ROOT_DIR, "results\\fcn_*")):
        id = int(file_path.split("\\")[-1][offset:])
        source_epochs_loss = join(file_path, "epochs_loss.png")
        source_parameters = join(file_path, "parameters.json")
        target_epochs_loss = join(TARGET_DIR, "epochs_loss_{}.png".format(id))
        target_parameters = join(TARGET_DIR, "parameters_{}.json".format(id))

        copyfile(source_epochs_loss, target_epochs_loss)        
        copyfile(source_parameters, target_parameters)