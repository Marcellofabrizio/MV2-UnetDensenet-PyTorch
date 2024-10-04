import os
import SimpleITK as sitk
import numpy as np
import json
from PIL import Image
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, Spacingd, ScaleIntensityd

def save_slice_as_png(slice_array, output_path):
    slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array)) * 255
    slice_array = slice_array.astype(np.uint8)
    image = Image.fromarray(slice_array)
    image.save(output_path)

def process_nifti_files(image_dir, label_dir, output_dir, json_file):
    os.makedirs(output_dir, exist_ok=True)
    
    output_info = []

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".nii.gz")])

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(224, 224, 131), mode=("area", "nearest")),
        ScaleIntensityd(
            keys=["image"],minv = 0.0, maxv = 1.0, factor = None
        )
    ])

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

            save_slice_as_png(image_vol[:, :, i], image_slice_path)
            save_slice_as_png(label_vol[:, :, i], label_slice_path)

            output_info.append({
                "image_path": image_slice_path,
                "label_path": label_slice_path,
                "original_image": image_path,
                "original_label": label_path
            })

    dataset = dict()
    dataset['description'] = 'mv2 dataset'
    dataset['labels'] = {
        "0": "background",
        "1": "mv2"
    }
    dataset['data'] = output_info

    with open(json_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Processing complete. Data saved to '{json_file}'.")

if __name__ == "__main__":
    image_dir = "/home/marcello/Documentos/dicoms/dicoms-dataset2/images"
    label_dir = "/home/marcello/Documentos/dicoms/dicoms-dataset2/labels"
    output_dir = "/home/marcello/Documentos/dicoms/dataset2"
    json_output_file = "/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset2/dataset2.json"

    process_nifti_files(image_dir, label_dir, output_dir, json_output_file)
