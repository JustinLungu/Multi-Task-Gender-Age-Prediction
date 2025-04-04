import pandas as pd
import os
# Function to rename files in a directory
def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg.chip.jpg"):
            # Create the new filename by replacing '.jpg.chip.jpg' with '.jpg'
            new_filename = filename.replace('.jpg.chip.jpg', '.jpg')

            # Construct the full file paths
            current_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(current_filepath, new_filepath)
            #print(f"Renamed {filename} to {new_filename}")

# Function to extract information from image file names
def extract_info_from_filename(filename):
    # Split the filename by underscores
    parts = filename.split('_')
    # Extract relevant information
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])
    return age, gender, race

def can_extract_info(filename):
    parts = filename.split('_')
    if (len(parts)<4):
        print(filename + " has missing values.")
        return False
    else:
        return True

# Function to get image file names in a folder
def get_image_info(folder_path):
    image_info = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Consider only JPG files
            if can_extract_info(filename):
                age, gender, race = extract_info_from_filename(filename)
                image_id = filename  # Remove file extension to get image ID
                image_info.append((image_id, age, gender, race))

    return image_info

# Create a DataFrame from image information
def create_dataframe(folder_path):
    image_info = get_image_info(folder_path)
    df = pd.DataFrame(image_info, columns=['Image_ID', 'Age', 'Gender', 'Race'])
    return df

def preprocess(folder_path, output_directory):
    rename_files(folder_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the output file path
    output_file_path = os.path.join(output_directory, 'UTKFace_labels.csv')

    # Save the DataFrame to a CSV file in the new directory
    image_dataframe = create_dataframe(folder_path)
    image_dataframe.to_csv(output_file_path, index=False)