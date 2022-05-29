from clearml import Task, Dataset, StorageManager
from pathlib import Path


def main():
    download_path = (
        "http://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz"
    )
    path_to_data = StorageManager.get_local_copy(download_path, extract_archive=True)
    path_to_data = Path(path_to_data) / 'empatheticdialogues'

    # Create a Dataset in ClearML
    clearml_dataset = Dataset.create(
        dataset_name='clearmldata', dataset_project='ParlAI'
    )

    clearml_dataset.add_files(path_to_data)
    clearml_dataset.finalize(auto_upload=True)


if __name__ == "__main__":
    clearml_task = Task.init(
        project_name='ParlAI', task_name='Download and Upload Data'
    )
    main()
