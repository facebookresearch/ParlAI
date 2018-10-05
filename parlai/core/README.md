parlai/core contains several files:

- **agents.py**: provides a set of basic agents and utility methods
- **build_data.py**: utilities for downloading and building data
- **dict.py**: code for parsing and building a dictionary from text
- **image_featurizers.py**: utilities for loading images and loading/extracting image features
- **logs.py**: interface to log any metrics in tensorboard, could be extended to any other tool like visdom
- **metrics.py**: provides standard metric evaluations for dialogue
- **params.py**: provides an argument parser as well as a set of default command line options for using the ParlAI package
- **pytorch_data_teacher.py**: dataloader which utilizes multiprocessed dataloading for streaming data from disk (rather than loading it into memory)
- **teachers.py**: provides a set of teachers that deal with dialog; also includes a threadpool data loader
- **thread_utils.py**: contains useful utilities for multiprocessing
- **torch_agent.py**: contains general utility code for building PyTorch-based agents in ParlAI
- **utils.py**: file for miscellaneous utility functions and constants
- **worlds.py**: provides a set of worlds, which define basic environments that control how agents interact with each other
