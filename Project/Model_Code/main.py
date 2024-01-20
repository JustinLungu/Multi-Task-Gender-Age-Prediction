from data_preprocessing import preprocess
import model

# True if you already have csv files of the split data in ../data/datasets/
# False if you want these files to be created
PREPROCESSED = True
#1 = 100% and 0 = 0%
DATA_PERCENTAGE = 0.1

if __name__ == "__main__":
    if PREPROCESSED == False:
        image_folder_path = '../data/UTKFace'
        df_output_directory = '../data'
        preprocess(image_folder_path, df_output_directory)

    # model_type determines the model type to be trained (surprise surprise)
    # can be 'final', 'age_baseline' and 'gender_baseline'
    model_type = 'gender_baseline'
    model.run_model(model_type, DATA_PERCENTAGE, PREPROCESSED)
