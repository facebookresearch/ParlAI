# Crowdsourcing utilities

## Overview
- `acceptability.py`: Used to ensure that a worker's messages throughout a conversation meet certain criteria (not too short, not all caps, not a lot of repetition, safety, etc.). More details about the acceptability checker can be found in the `acceptability.AcceptabilityChecker` section below.
- `analysis.py`: Abstract base classes for compiling the results of crowdsourcing runs.
- `frontend.py`: Method for compiling the frontend code of crowdsourcing tasks.
- `mturk.py`: Code for soft-blocking MTurk crowdsourcing workers (preventing them from working on this specific task), as well as a Hydra flag to pass in lists of workers to soft-block.
- `tests.py`: Abstract base classes for testing different categories of crowdsourcing tasks.
- `worlds.py`: Abstract base classes for onboarding and chat worlds.

## `acceptability.AcceptabilityChecker`

### How to add a new check
- Add the code for this in `.check_messages()`, inside a `if 'check_name' in violation_types:` condition
- Add the name of the check to `self.ALL_VIOLATION_TYPES`; otherwise, this check will not be recognized, and an error will be raised if the check is specified when calling `.check_messages()`!
- To use the check: add the name of the check to the `violation_types` arg when calling `.check_messages()`
