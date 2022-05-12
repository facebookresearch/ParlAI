# DARMA model chat 

Main command: `python run.py`
This will use the default configuration of `conf=darma` 

- Baseline setup: `python run.py conf=example`. 
- Refer to https://github.com/facebookresearch/ParlAI/tree/main/model_chat

## General tips 

If you don't know where to start, run the baseline example with `python run.py conf=example` and see in the terminal all the parameters that were used for this run. Most of these parameters are well-explained in `model_chat_blueprint.py`. 


### Frontend Customizations

Here, we map frontend customizations and the corresponding scripts that need to be modified. 
- left pane instructions: 
  - `task_config/darma_task_description.html`
- task description: 
  - `task_config/darma_left_pane_text.html`
- title: 
  - simply update `task_title` in `darma.yaml`
- Post-conversation survey: 
  - Selection choices: 
    - `parlai/crowdsourcing/tasks/darma_chat/frontend/components/response_panes.jsx` under `function RatingSelector`
  - Survey questions: 
    - set `final_rating_question` with your question. For multiple questions, separate them by "|" in a single string. 
- Onboarding task (informed consent form): 
  - Update configuration in `darma.yaml`
    - onboard_task_data_path: `${task_dir}/task_config/darma_onboard_task_data.json`
    - annotations_config_path:`${task_dir}/task_config/darma_annotations_config.json`
  - Description: 
    - `frontend/components/onboarding_components.jsx` under `<OnboardingDirections>` 
  - Checkbox: 
    - Specify onboarding questions and onboarding requirement: `task_config/darma_onboard_task_data.json`
    - Answer choices: `task_config/darma_annotations_config.json`
  - TODO: import from `task_config/informed_consent.html` to keep code more organized. 

## Seed conversation customizations 

- By setting `conversation_start_mode: custom` in `darma.yaml` and specifying the path to a json file with `seed_conversation_source`, we load a json file that contain dialogues that can seed conversations for evaluation. 
- modify the `_run_initial_turn` for `ModelChatWorld`  in `worlds.py` to load dialogue data and establish it as the starting point. 

## Model customization

- Specify the model to chat with in `task_config/darma_model_opts`
  - choose either a model that is hosted by ParlAI or provide a path to a model that has been trained locally with ParlAI. 

## Debug logs/tips 

- Q: I'm making changes to the front end and it seems like they are not reflected in my task. 
  - A: examine the config that gets printed when running `python run.py` and make sure that all configurations point to the directory that you're making changes to. The `get_task_path()` function and imports that have not been updated after copying files from another crowdsourcing task directory may be the culprit. 
- To use custom opts (configs) in `worlds.py`, make sure to add them to the blueprint arguments, and then add them where `shared_state.world_opt.update` is called. 
