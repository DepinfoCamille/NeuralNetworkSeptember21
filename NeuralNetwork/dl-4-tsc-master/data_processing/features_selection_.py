from shutil import copyfile, copytree
from glob import glob
from os.path import join

REPAIR_PATH = True

ROOT_DIR = "C:\\Users\\Camille\\Documents\\These\\ExperienceSeptember21\\NeuralNetwork\\dl-4-tsc-master\\tune_parameters"
TARGET_DIR = join(ROOT_DIR, "results_summary")
if REPAIR_PATH:
    ROOT_DIR = "/workspace/blabla"
    TARGET_DIR = join(RROOT_DIR, "results")
if __name__ == "__main__":

    if REPAIR_PATH:
        for file_path in glob(join(ROOT_DIR, "results\\\\fcn_*")):
            input_local_filepath = file_path.split("\\\\")[-1]
            output_filepath = join(TARGET_DIR, input_local_filepath)
            
            copytree(file_path, output_filepath)        

    else:

        offset = len("fcn_multi_labels_")
        for file_path in glob(join(ROOT_DIR, "results\\fcn_*")):
            id = int(file_path.split("\\")[-1][offset:])
            source_epochs_loss = join(file_path, "epochs_loss.png")
            source_parameters = join(file_path, "parameters.json")
            target_epochs_loss = join(TARGET_DIR, "epochs_loss_{}.png".format(id))
            target_parameters = join(TARGET_DIR, "parameters_{}.json".format(id))

            copyfile(source_epochs_loss, target_epochs_loss)        
            copyfile(source_parameters, target_parameters)

