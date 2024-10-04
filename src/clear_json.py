import json
import os

with open('/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset2/dataset2.json', 'r') as file:
    data_dict = json.load(file)

# Filter the data by checking if both image and label files exist on disk
filtered_data = [
    item for item in data_dict['data']
    if os.path.exists(item['image_path']) and os.path.exists(item['label_path'])
]

# Update the 'data' key in the original dictionary with filtered data
data_dict['data'] = filtered_data

# Save the updated JSON back to the file (or a new file)
with open('filtered_json_file.json', 'w') as file:
    json.dump(data_dict, file, indent=4)

print(f"Filtered dataset saved successfully. {len(filtered_data)} items remain.")