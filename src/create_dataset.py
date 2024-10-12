import os
import sys
import SimpleITK as sitk
import numpy as np
import json
from PIL import Image
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, Spacingd, ScaleIntensityd

def to_np_array(slice_array):
    slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array)) * 255
    return slice_array.astype(np.uint8)

def save_slices(image_array, label_array, image_path, label_path, exclusion_chance):
    image_np_array = to_np_array(image_array)
    label_np_array = to_np_array(label_array)
    
    if np.all(label_np_array == 0):
        if np.random.rand() <= exclusion_chance:
            print(f"{label_path} is empty. Excluding from dataset")
            return False
    
    image = Image.fromarray(image_np_array)
    image.save(image_path)
    
    label = Image.fromarray(label_np_array)
    label.save(label_path)
    
    return True

def create_images(files, base_dir, output_dir, exclusion_chance):
    
    image_dir = f"{base_dir}/images"
    label_dir = f"{base_dir}/labels"
    
    image_files = []
    label_files = []
    
    for file in files:
        image_files.append(file["image"])
        label_files.append(file["label"])

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(224, 224, 131), mode=("area", "nearest")),
        ScaleIntensityd(
            keys=["image"],minv = 0.0, maxv = 1.0, factor = None
        )
    ])
    
    output_info = []
    
    for image_file, label_file in zip(image_files, label_files):
        print(f"Processing '{image_file}' and '{label_file}'...")

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        data = {"image": image_path, "label": label_path}

        transformed_data = transforms(data)

        image_vol = transformed_data["image"][0]
        label_vol = transformed_data["label"][0]

        for i in range(image_vol.shape[2]):
            file_name = str(image_file).rsplit('.nii.gz', 1)[0]

            image_slice_filename = f"{os.path.splitext(file_name)[0]}_slice_{i}.png"
            label_slice_filename = f"{os.path.splitext(file_name)[0]}_slice_{i}.png"

            image_slice_path = os.path.join(output_dir, "images", f"{image_slice_filename}")
            label_slice_path = os.path.join(output_dir, "labels", f"{label_slice_filename}")

            has_saved = save_slices(
                image_vol[:, :, i], 
                label_vol[:, :, i],
                image_slice_path,
                label_slice_path,
                exclusion_chance
                )

            if has_saved:
                output_info.append({
                    "image": image_slice_filename,
                    "label": label_slice_filename,
                    "original_image": image_path,
                    "original_label": label_path
                })
    
    return output_info

def process_nifti_files(train_files, test_files, base_dir, output_dir, json_file, exclusion_chance=0.0):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    dataset = dict()
    dataset['description'] = 'MV2 dataset'
    dataset['labels'] = {
        "0": "background",
        "1": "mv2"
    }

    dataset['data'] = {
        "train": [],
        "test": []
    }

    train_images = create_images(train_files, base_dir, output_dir, exclusion_chance)
    test_images = create_images(test_files, base_dir, output_dir, exclusion_chance)

    dataset['trainImages'] = len(train_images)
    dataset['testImages'] = len(test_images)

    dataset['imageOutputPath'] = os.path.join(output_dir, "images")
    dataset['labelOutputPath'] = os.path.join(output_dir, "labels")

    dataset['data']["train"] = train_images
    dataset['data']["test"] = test_images

    with open(json_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Processing complete. Data saved to '{json_file}'.")

if __name__ == "__main__":
    
    dicom_definition_json_path = sys.argv[1]
    exclude_blank_percentage = float(sys.argv[2])
    
    with open(dicom_definition_json_path, 'r') as file:
        dicom_definition_json = json.load(file)

    train_files = dicom_definition_json["data"]["train"]
    test_files = dicom_definition_json["data"]["test"]

    base_dir = dicom_definition_json["basePath"]
    
    output_dir = "/home/marcello/Documentos/dicoms/dataset4"
    json_output_file = "/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset4/dataset4.json"

    process_nifti_files(train_files, test_files, base_dir, output_dir, json_output_file, exclude_blank_percentage)