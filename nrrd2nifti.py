import SimpleITK as sitk
import sys
import os

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_nrrd_file>")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.isfile(input_file):
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)

img = sitk.ReadImage(input_file)

output_file = os.path.splitext(input_file)[0] + ".nii.gz"

sitk.WriteImage(img, output_file)

print(f"Successfully converted '{input_file}' to '{output_file}'.")