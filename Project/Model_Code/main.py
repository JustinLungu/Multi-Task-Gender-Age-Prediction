from data_preprocessing import preprocess
import model
import os

# True if you already have csv files of the split data in ../data/datasets/
# False if you want these files to be created
PREPROCESSED = False
#1 = 100% and 0 = 0%
DATA_PERCENTAGE = 1
EPOCHS = 10
BATCH_SIZE = 32
# model_type determines the model type to be trained (surprise surprise)
# can be 'final', 'age_baseline' and 'gender_baseline'
MODEL_TYPE = 'gender_baseline'

if __name__ == "__main__":
    # current working directory
    current_directory = os.getcwd()
    print("Directory: ", current_directory)

    if PREPROCESSED == False:
        image_folder_path = os.path.join(current_directory, 'Data', 'UTKFace')
        df_output_directory = os.path.join(current_directory, 'Data')
        preprocess(image_folder_path, df_output_directory)

    model.run_model(MODEL_TYPE, DATA_PERCENTAGE, PREPROCESSED, EPOCHS, BATCH_SIZE)
