import json
import os

with open('/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset4/dataset4_25.json', 'r') as file:
    data_dict = json.load(file)

# Filter the data by checking if both image and label files exist on disk
filtered_train_data = [
    item for item in data_dict['data']["train"]
        if os.path.exists(os.path.join(data_dict['imageOutputPath'], item['image'])) 
            and os.path.exists(os.path.join(data_dict['labelOutputPath'], item['label']))
]

filtered_test_data = [
    item for item in data_dict['data']["test"]
        if os.path.exists(os.path.join(data_dict['imageOutputPath'], item['image'])) 
            and os.path.exists(os.path.join(data_dict['labelOutputPath'], item['label']))
]

# Update the 'data' key in the original dictionary with filtered data
data_dict['data']["train"] = filtered_train_data
data_dict['data']["test"] = filtered_test_data

data_dict['trainImages'] = len(data_dict['data']["train"])
data_dict['testImages'] = len(data_dict['data']["test"])

# Save the updated JSON back to the file (or a new file)
with open('/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset4/dataset4_25.json', 'w') as file:
    json.dump(data_dict, file, indent=4)
