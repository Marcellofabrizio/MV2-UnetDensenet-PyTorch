import json
import os
from PIL import Image
import numpy as np

# Function to check if an image is all zeros
def is_label_all_zeroes(label_path):
    try:
        # Open the label image
        with Image.open(label_path) as img:
            img_array = np.array(img)  # Convert the image to a NumPy array

        # Check if all elements in the image are zero
        return np.all(img_array == 0)
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return False

# Function to delete both image and label files if label is all zeroes
def remove_zero_label_entries_and_delete_files(data):
    filtered_data = []
    for item in data:
        label_path = item.get('label')
        image_path = item.get('image')
        
        # Check if both image and label paths exist
        if os.path.exists(label_path) and os.path.exists(image_path):
            if is_label_all_zeroes(label_path):
                # If the label is all zeroes, delete both the image and label files
                try:
                    os.remove(label_path)
                    os.remove(image_path)
                    print(f"Deleted: {label_path} and {image_path}")
                except Exception as e:
                    print(f"Error deleting files {label_path} or {image_path}: {e}")
            else:
                # Keep the entry if the label is not all zeroes
                filtered_data.append(item)
        else:
            print(f"Label or image file {label_path} or {image_path} does not exist, skipping this entry.")
    
    return filtered_data

# Load the JSON dataset
with open('/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset3/dataset3.json', 'r') as file:
    dataset = json.load(file)

# Filter the data array and delete files with all-zero labels
filtered_data = remove_zero_label_entries_and_delete_files(dataset['data'])

# Update the dataset with filtered data
dataset['data'] = filtered_data

# Save the updated dataset to a new file or overwrite the existing one
with open('filtered_dataset.json', 'w') as file:
    json.dump(dataset, file, indent=4)

print(f"Filtered dataset saved successfully. {len(filtered_data)} items remain.")