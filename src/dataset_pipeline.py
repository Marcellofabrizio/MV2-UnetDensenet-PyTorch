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

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".nii") or f.endswith(".seg.nii.gz")])

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=(160, 160, 131), mode=("area", "nearest")),
        ScaleIntensityd(keys=["image"])
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
            image_slice_filename = f"{os.path.splitext(image_file)[0]}_slice_{i}.png"
            label_slice_filename = f"{os.path.splitext(label_file)[0]}_slice_{i}.png"

            image_slice_path = os.path.join(output_dir, f"img_{image_slice_filename}")
            label_slice_path = os.path.join(output_dir, f"seg_{label_slice_filename}")

            save_slice_as_png(image_vol[:, :, i], image_slice_path)
            save_slice_as_png(label_vol[:, :, i], label_slice_path)

            output_info.append({
                "image_path": image_slice_path,
                "label_path": label_slice_path,
                "original_image": image_path,
                "original_label": label_path
            })

    with open(json_file, 'w') as f:
        json.dump(output_info, f, indent=4)

    print(f"Processing complete. Data saved to '{json_file}'.")

if __name__ == "__main__":
    image_dir = "/home/marcello/Documentos/Universidade/Dicoms/cropped-dicoms/images"
    label_dir = "/home/marcello/Documentos/Universidade/Dicoms/cropped-dicoms/labels"
    output_dir = "/home/marcello/Documentos/dicoms/dataset1"
    json_output_file = "/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset1/dataset1.json"

    process_nifti_files(image_dir, label_dir, output_dir, json_output_file)
