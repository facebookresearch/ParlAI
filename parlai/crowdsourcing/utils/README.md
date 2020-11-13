# Crowdsourcing utilities

## `acceptability.AcceptabilityChecker`

* Used to make sure that a worker's messages throughout a conversation meet certain criteria (not too short, not all caps, not a lot of repetition, safety, etc.)
### How to add a new check
- Add the code for this in `.check_messages()`, inside a `if 'check_name' in violation_types:` condition
- Add the name of the check to `self.possible_violation_types`; otherwise, this check will not be recognized, and an error will be raised if the check is specified when calling `.check_messages()`!
- To use the check: add the name of the check to the `violation_types` arg when calling `.check_messages()`
