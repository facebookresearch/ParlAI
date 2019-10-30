# Setting up ParlAI for a custom chat-service

## Message Format
To maintain consistency we are trying to enforce a deterministic message format throughout this task. If a particular chat service doesn't adhere to this format, one must use ```restructure_message()``` to adapt the messages to this format before the messages being used. The message format is defined below:
```python
# TODO
```

## Config
Below is the standard config format to be followed across all chat services:
```
tasks:
  default:
    onboard_world: # Same as earlier
    task_world: # Same as earlier
    timeout: # Same as earlier
    agents_required: # Same as earlier
task_name: # Same as earlier
world_module: # Same as earlier
overworld: # Same as earlier
max_workers: # Same as earlier
opt:  # Additional model opts go here
  debug: # Same as earlier
  model: # Same as earlier
  model_file: # Same as earlier
  override:
    model: # Same as earlier
additional_args:  # Additional chat service specific args go here
  page_id: 1 # Facebook Page id (if Messenger, else don't include this field)
  *any other args needed by <chat_service>*
```
As one can notice, most of the format is the same as how it already exists for messenger with the exception of having ```additional_args:``` as a field in our config. This has been introduced to provide flexibility of parsing any additional arguments a chat service may need whilst preserving the previously existing necessary args. Note however that ```page_id``` has been shifted to this section to maintain coherence.

## Manager
### Socket, Runner etc.

## Agents
