# Setting up ParlAI for a custom chat-service
This is an instruction manual to be used as reference on how to configure ParlAI for an arbitrary chat service.

## Message Format
To maintain consistency we are trying to enforce a deterministic message format throughout this task. If a particular chat service doesn't adhere to this format, one must use ```restructure_message()``` to adapt the messages to this format before the messages are used. The message format is defined below:
```config
{
  mid: # ID of this message
  recipient: {
    id: #id of message recipient
  }
  sender: {
    id: #id of message sender
  }
  text: # text of the message
  attachment: # attachment of the message
}
```
### Additional Message fields
These define a non-exhaustive list of keys that one could use in the message dict for ease-of-use
- messaging_type: whether the message is a text message or an image upload [RESPONSE, UPDATE]
- quickreplies: Auto-suggested replies
- persona_id: id of the persona that is interacting
- name: Display name of the user
- profile_picture_url: URL to the profile picture of the user


## Config
Below is the standard config format and hierarchy to be followed across all chat services:

```config
- tasks:  # List of available tasks/worlds for a user to enter
    - <task 1 name>
      - onboard_world: # World in which user is first send upon selecting the task. Can collect necessary data from user in this world, as well as provide instructions etc.
      - task_world: # Actual task world
      - timeout: # Agent message timeout - how long to wait for the agent to send a message before assuming they have disconnected.
      - agents_required: # Number of agents required to run the task world, e.g. 2 for a two player game, 1 for a one-player experience, etc.
    - <task 2 name>
      - onboard_world:
      - task_world:
      - timeout:
      - agents_required:
- task_name: # name of the overall task
- world_module: # module in which all of the worlds exist (relative path i.e. `parlai.chat_service....`
- overworld: # Name of the overworld; where the agent is first sent upon messaging the service
- max_workers: # Maximum number of workers that can be in task worlds at any given moment
- opt:  # Additional model opts go here. Below are example opts that one could normally pass to parlai
    - password: # Password for messaging service, if this is wanted
    - debug: # whether to set debug mode
    - model: # Name of model, if you want to load a model
    - model_file: # path to model file, if you want to load a model
    - override:
        - model: # overrides for model
- additional_args:  # Additional chat service specific args go here
    - service_reference_id: 1 # Facebook Page id (if Messenger, else don't include this field)
    -  *any other args needed by <chat_service>*
```

As one can notice, most of the format is the same as how it already exists for messenger with the exception of having ```additional_args:``` as a field in our config. This has been introduced to provide flexibility of parsing any additional arguments a chat service may need whilst preserving the previously existing necessary args. Note however that ```page_id``` has been shifted to this section to maintain coherence.

## Manager

To implement your own service, you will need to subclass the ```ChatServiceManager``` in ```chat_service/core/chat_service_manager.py```. The ```ChatServiceManager``` handles a lot of the abstraction for you, so you'll only need to subclass a few of the methods in there when implementing your own service.

Some notable essentials below (note not all abstract methods are specified):

1. ```ChatServiceMessageSender``` - The message sender is useful for wrapping requests with additional functions. E.g., in the Messenger service, the ```MessageSender``` implements ```send_read``` and ```typing_on``` functions which are called upon message sends.
2. ```parse_additional_args``` - This is the function for parsing the additional args from the config specified above
3. ```_complete_setup``` - Complete any additional setup items; should be called in initialization.
4. ```_load_model``` - This function should load the model, if necessary. This varies per chat service.
5. ```restructure_message``` - If messages in your service are not the same format as specified in *Message Format* above, please use this method to restructure the message.
6. ```setup_server``` - sets up the chat service server.
7. ```setup_socket``` - sets up the socket to start communicating with users.
8. ```observe_message``` - Send a message thru the message manager to a corresponding user/agent.


## Manager Utils
### Socket

The ```ChatServiceMessageSocket``` is a wrapper around websockets to forward messages from a remote server to a ```ChatServiceManager```.

### Runner

The ```ChatServiceWorldRunner``` is the actual class for handling running worlds, overworlds, etc.

## Agents

Finally, once you have implemented your ```ChatServiceManager```, you will need to implement a ```ChatServiceAgent```, which is specific for each chat service. The following are notable functions to implement in your own ```ChatServiceAgent```.

1. ```observe``` - Implement this function to receive messages from the manager.
2. ```put_data``` - Puts data into the Agent's message queue if it hasn't already been seen.
