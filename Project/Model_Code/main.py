from data_preprocessing import preprocess
import model

PREPROCESSED = True

if __name__ == "__main__":
    if PREPROCESSED == False:
        image_folder_path = '../data/UTKFace'
        df_output_directory = '../data'
        preprocess(image_folder_path, df_output_directory)

    # model_type determines the model type to be trained (surprise surprise)
    # can be 'final', 'age_baseline' and 'gender_baseline'
    model_type = 'gender_baseline'
    model.run_model(model_type)
