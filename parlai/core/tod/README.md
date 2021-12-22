# Task-Oriented Dialog (TOD) Core README

For the quickest getting-to-use of TOD classes, start with the "Teachers + Agents Usage" section below (for understanding how to setup agents such that they work with new datasets) and `parlai/scripts/tod_world_script.py` (for understanding how to run simulations with the TOD conversations format). 

See `projects/tod_simulator/README` for a higher-level usage-focused README. This document also describes the structure of the contents of this directory. 

As a convention, files referenced externally to this directory are prefixed with `tod` whereas those only referenced by other files within the directory are not. 

---

# Teachers + Agents Usage

tl;dr Extend `TodStructuredDataParser` for your particular dataset and implement `setup_episodes()` that converts the dataset into a list of episodes (`List[TodStructuredEpisode]`). Use multiple inheritence to generate teachers for training models. See files like `parlai/tasks/multiwoz_v22/agents.py` for an example. 

See `tod_agents.py` for the classes.  

## Overview of usage

For a given dataset, extend `TodStructuredDataParser` and implement `setup_episodes()` and `get_id_task_prefix()`. The former of these is expected to do the data processing to convert a dataset to `List[TodStructuredEpisode]`. From here, multiple inheritance can be used to define Agents and Teachers that utilize the data.

For example, given a `class XX_DataParser(TodStructuredDataParser)`, `class XX_UserSimulatorTeacher(XX_DataParser, TodUserSimulatorTeacher)` would be how one would define a teacher that generates training data for a User Simulator model. Other agents necessary for exposing Goals and API Schemas of a dataset can be exposed in a similar manner. 

See `tod_agents.py` for the classes.  

Once the relevant agents have been created (or relevant models have been fine-tuned), see `parlai.scripts.tod_world_script` for generating the simulations themselves.

## Why we do this
These files aid in consistency between Teachers and Agents for simulation. Rather than having to align multiple different agents to be consistent about assuptions about data formatting, tokens, spacing, etc, we do this once (via converting everything to `TodStructuredEpisode`) and let the code handle the rest.

# Description of Agents + Teachers useful for Simulation
## Teachers for training (generative) models
    * TodSystemTeacher
    * TodUserSimulatorTeacher

## Agents for Grounding
For goal grounding for the User for simulation:
    * TodGoalAgent
        * Dumps goals as is from the dataset, possibly multiple per episode
    * TodSingleGoalAgent
        * Flattens goals such that only one is used to ground a conversation. (So for example, if there are 3 goals in the source dataset, rather than grounding 1 conversation, these now ground 3 separate conversations.)

For (optional) API schema grounding for the System:
    * TodApiSchemaAgent (must be used with `TodGoalAgent` only)
    * TodSingleApiSchemaAgent (must be used with `TodSingleGoalAgent` only)
    * EmptyApiSchemaAgent
        * Used for simulations where the expectation is `no schema`, ie, evaluation simulations.

## Agents for mocking APIs:
    * StandaloneApiAgent
         * Assumed to be provided a .pickle file 'trained' by `TodStandaloneApiTeacher`. (See comments in-line on classes for train command example)

# Agents for dumping data from a ground truth dataset
The following are for extracting TOD World metrics from a ground truth dataset. These are generally used sparingly and only for calculating baselines.
    * TodApiCallAndSysUttAgent
    * TodApiResponseAgent
    * TodUserUttAgent

For this metrics extraction, `TodGoalAgent` and `TodApiSchemaAgent` should be used.

# Other agents
There is a `EmptyGoalAgent` for use in human-human conversations where a goal is unnecessary.

---

# Directory contents

This directory is split into 3 main components: files to support agents + teachers, files to support the simulation world, and files to store functionality common to both of these. We describe the common functionality first then go to the other two.

Tests for all files in this directory are stored in `tests/tod`

## Files for common functionality 
`tod_core.py` defines consts and enums used across TOD agents, teachers, and world. It also defines dataclasses for storing the intermediate data format used when parsing a dataset to the TOD structure as well as a `SerializationHelper` from going from machine structured data (ex. API Calls) to flattened versions used by the models.


## Files for agents and teachers
Usage of `tod_agents.py` is described above. It references `teacher_metrics.py` which stores Metrics objects.

## Files for simulation world
Description of usage of the simulation world is primarily stored in the script running the world, stored in `parlai/scripts/tod_world_script.py`. The script is responsible for running multiple episodes of simulation and saving simulation output data. 

The world itself is stored in `tod_world.py`. The world follows the same intermediate dataformats for episodes as described in `tod_core.py` and does the correct calling of different agents to support this. It is generally recommended that this file not be touched. 

A general class for collecting metrics out of `TODWorld` is stored within `world_metrics.py` with individual 'metric handlers' responsible for calculating a given metric stored in `world_metric_handlers.py`. 

