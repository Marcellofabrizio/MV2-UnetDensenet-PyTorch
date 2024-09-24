import SimpleITK as sitk
import sys
import os

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_folder>")
    sys.exit(1)

input_folder = sys.argv[1]

if not os.path.isdir(input_folder):
    print(f"Error: Directory '{input_folder}' not found.")
    sys.exit(1)

for filename in os.listdir(input_folder):
    if filename.endswith(".nrrd"):
        input_file = os.path.join(input_folder, filename)
        print(f"Processing '{input_file}'...")

        try:
            img = sitk.ReadImage(input_file)

            output_file = os.path.splitext(input_file)[0] + ".nii.gz"

            sitk.WriteImage(img, output_file)

            print(f"Successfully converted '{input_file}' to '{output_file}'.")

        except Exception as e:
            print(f"Error processing file '{input_file}': {e}")
    else:
        print(f"Skipping non-NRRD file: '{filename}'")

print("All files processed.")