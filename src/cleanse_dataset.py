import json
import os

# Machine God, cleanse the dataset from the Archenemy interference. Praise be The Omnissiah

json_path = '/home/marcello/Repositories/DICOM-Project-Pytorch/data/dataset10/dataset10.json'

with open(json_path, 'r') as file:
    data_dict = json.load(file)

filtered_train_data = []
filtered_test_data = []

for item in data_dict['data']['train']:
    image = os.path.join(data_dict['imageOutputPath'], item['image'])
    label = os.path.join(data_dict['labelOutputPath'], item['label'])
    if (not os.path.exists(image)) and os.path.exists(label):
        os.remove(label)
    else:
        filtered_train_data.append(item)

for item in data_dict['data']['test']:
    image = os.path.join(data_dict['imageOutputPath'], item['image'])
    label = os.path.join(data_dict['labelOutputPath'], item['label'])
    if (not os.path.exists(image)) and os.path.exists(label):
        os.remove(label)
    else:
        filtered_test_data.append(item)

data_dict['data']["train"] = filtered_train_data
data_dict['data']["test"] = filtered_test_data

data_dict['trainImages'] = len(data_dict['data']["train"])
data_dict['testImages'] = len(data_dict['data']["test"])


with open(json_path, 'w') as file:
    json.dump(data_dict, file, indent=4)
