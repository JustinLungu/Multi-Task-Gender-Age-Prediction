from data_preprocessing import preprocess
import model

PREPROCESSED = True

if __name__ == "__main__":
    if PREPROCESSED == False:
        image_folder_path = '../data/UTKFace'
        df_output_directory = '../data'
        preprocess(image_folder_path, df_output_directory)

    model.run_model()
