# Dialogue Safety Tests

A suite of dialogue safety unit tests and integration tests, in correspondence with the paper <TODO: PAPER LINK>

## Paper Information
TODO: fill me in

**Abstract:** TODO: write me


## Setting up the API
The tests require *only* implementing only the following API:
```
def get_response(self, input_text: str) -> str:
```
This function takes as input the dialogue history (`input_text`) and returns the dialogue model's response (as a string).

> NOTE: One does not need to implement a ParlAI agent to run these unit tests; the API only requires text in, text out.

One must add one's model wrapper to the folder `projects/safety_bench/model_wrappers` and register it via `@register_model_wrapper("model_name")` so that it is accessible on the command line.

## Unit Tests

The unit tests run automatically provided the above API access to the model.

Details on these tests can be found in Section 6 of the paper. We test both:
1. The model's ability to generate offensive language and
2. How the model responds to offensive language.

### Example commands

Run unit tests for the model `blenderbot_90M` and safe all logs to the folder `/tmp/blender90M`:
```
python projects/safety_bench/run_unit_tests.py --wrapper blenderbot_90M --log-folder /tmp/blender90M
```

Run unit tests for the model `gpt2_large` and safe all logs to the folder `/tmp/gpt2large`:
```
python projects/safety_bench/run_unit_tests.py -w gpt2_large --log-folder /tmp/gpt2large
```

## Integration Tests
Provided the same API access as described above, we provide tooling to make it easy to run the human safety evaluations on Mechanical Turk from [here](https://parl.ai/projects/dialogue_safety/).

These tools prepare data as input for the Mechanical Task. Further instructions for setting up [Mephisto](https://github.com/facebookresearch/Mephisto) and running the task on Mechanical Turk are printed at the completion of the script.

### Example Commands
Prepare integration tests for the adversarial setting for the model `blenderbot_3B`:
```
python projects/safety_bench/prepare_integration_tests.py --wrapper blenderbot_3B --safety-setting adversarial
```

Prepare integration tests for the nonadversarial setting for the model `dialogpt_medium`:
```
python projects/safety_bench/prepare_integration_tests.py --wrapper dialogpt_medium --safety-setting nonadversarial
```