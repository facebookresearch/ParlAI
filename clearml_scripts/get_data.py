from clearml import Dataset
import os

def get_data_from_clearml(opt):
      dataset_name = 'empatheticdialogues'
      clearml_dataset = Dataset.get(dataset_name= dataset_name, dataset_project="ParAI")
      dataset_localpath = os.path.join(opt['datapath'], dataset_name, dataset_name)
      clearml_dataset.get_mutable_local_copy(target_folder= dataset_localpath, overwrite=True)

if __name__ == "__main__":
  pass