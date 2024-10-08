import os

def delete_unmatched_images(images_dir, labels_dir):
    # Get a set of filenames (without the directory and extension) from the labels directory
    label_files = {os.path.basename(f) for f in os.listdir(labels_dir)}

    # Iterate through all files in the images directory
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)

        # Check if the corresponding file exists in the label directory
        if image_file not in label_files:
            print(f"Deleting {image_path}...")  # Informing about the file to be deleted
            os.remove(image_path)  # Delete the file

if __name__ == "__main__":
    # Set the directories
    images_dir = "/home/marcello/Documentos/dicoms/dataset3/images"  # Replace with the actual path to your images directory
    labels_dir = "/home/marcello/Documentos/dicoms/dataset3/labels"  # Replace with the actual path to your labels directory

    # Call the function to delete unmatched images
    delete_unmatched_images(images_dir, labels_dir)