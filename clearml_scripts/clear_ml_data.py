from clearml import Dataset

def main():
  dataset = Dataset.create(dataset_name='empatheticdialogues',
        dataset_project='ParAI'
)
  data_path = "/home/inan/Documents/GitHub/ParlAI/data/empatheticdialogues/empatheticdialogues"
  dataset.add_files(data_path)
  dataset.finalize(auto_upload=True)

if __name__ == "__main__":
  main()