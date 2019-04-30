## Included

`parlai/mturk/core` contains several files and folders that are needed for proper MTurk integration:

### Folders

- **react_server/**: contains the build files for running a task that uses a react frontend. This folder gets cloned, built, and then pushed to the hosting server (usually heroku).
- **tests/**: contains all of the tests for the MTurk functionality, split up by the various stages that they touch.

### Files

- **agents.py**: contains the `MTurkAgent` class, which is responsible for both acting as a proper agent in a regular ParlAI world while also keeping the relevant MTurk HIT assignment state.
- **data_model.py**: contains a few constants generally shared between the python backend and the javascript client that workers see.
- **mturk_data_handler.py**: class that manages a sqlite database that stores state and assignment data during and between runs. It also handles the local saving and loading of data for a task when handled using the use_db flag in `MTurkManager`.
- **mturk_manager.py**: contains the `MTurkManager` class, which is responsible for the lifecycle of an MTurk task as well as coordinating HITs and assignments to worlds and workers.
- **mturk_utils.py**: helper functions that directly interface with the MTurk API.
- **server_utils.py**: helper functions that are used to set up and take down the passthrough server that directly serves workers a task
- **shared_utils.py**: helper functions that are shared between multiple classes but don't actually relate enough to any of them to be a member of another file
- **socket_manager.py**: class that handles incoming and outgoing messages to workers as well as maintaining active connections with workers via websocket protocols
- **worker_manager.py**: class that is responsible for maintaining and acting directly on the state of workers as they complete tasks, disconnect, change worlds, and more.
- **worlds.py**: contains the basic onboarding and task worlds that should be used for MTurk tasks, as well as a description of how the underlying `MTurkDataWorld` handles saving data for a task.

## Created

parlai/mturk/core on your local machine may also create a few folders and files depending on your usage:

### Folders

- **heroku_server_.../**: contains the build files for a server that was started using ParlAI. If this remains after no servers are running, it exists because a server failed to cleanup the local files, and the folder can be deleted freely.
- **heroku-cli-v.../**: contains the bin files required to run heroku. These are stored here where we know where they are so that the server setup scripts don't need to search for existing heroku installations elsewhere.

### Files

- **disconnects.pickle**: a pickle file containing current outstanding disconnects. Used to track bad behavior from agents who have disconnected on multiple tasks. In the process of being deprecated in favor of storage in the state database (`pmt_data.db`) using the `use_db` flag.
- **working_time.pickle**: a pickle file containing the workers who have exceeded the max daily time across all of your tasks per day. In the process of being deprecated in favor of storage in the state database (`pmt_data.db`) using the `use_db` flag.
