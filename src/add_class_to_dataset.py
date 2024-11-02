import os
import sys
import json

if __name__ == "__main__":

    dicom_definition_json_path = sys.argv[1]
    dataset_json_path = sys.argv[2]
    
    with open(dicom_definition_json_path, "r") as file:
        dicom_definition_json = json.load(file)
    
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)

    
    updated_train = []
    updated_test = []
    
    for item in dicom_definition_json['data']['train']:
        file_name = item['image']
    
        for dt_item in dataset_json['data']['train']:
            if dt_item['original_image'] == file_name:
                new_item = dt_item
                dt_item['type'] = item['type']
                updated_train.append(new_item)
                
    
    for item in dicom_definition_json['data']['test']:
        file_name = item['image']
    
        for dt_item in dataset_json['data']['test']:
            if dt_item['original_image'] == file_name:
                new_item = dt_item
                dt_item['type'] = item['type']
                updated_test.append(new_item)

    with open(dataset_json_path, 'w') as file:
        json.dump(dataset_json, file, indent=4)
