# Tests in ParlAI

Author: Kurt Shuster

## Running a Test

To run tests in ParlAI, we use [pytest](https://docs.pytest.org/en/stable/). The following commands are taken from the `tests` directory [README](https://github.com/facebookresearch/ParlAI/tree/master/tests).

*To run all tests in your current directory, simply run:*
```bash
$ pytest
```

*To run tests from a specific file, run:*
```bash
$ pytest tests/test_tga.py
```

*To use name-based filtering to run tests, use the flag `-k`. For example, to only run tests with `TransformerRanker` in the name, run:*
```bash
pytest -k TransformerRanker
```

*For verbose testing logs, use `-v`:*
```bash
pytest -v -k TransformerRanker
```

*To print the output from a test or set of tests, use `-s`:*
```bash
pytest -s -k TransformerRanker
```

## Writing a Test

### Continuous Integration, Explained

So, you want to develop in ParlAI - awesome! We welcome any and all contributions. However, we always want to ensure that the platform remains stable and functioning, so we often encourage authors of pull requests to include a test with their changes. To ensure that our code remains correct and properly functioning, we use **continuous integration** - on each commit in a pull request, CircleCI will run a suite of tests that we've written to ensure that nothing breaks. Our CircleCI configuration can be found [here](https://github.com/facebookresearch/ParlAI/blob/master/.circleci/config.yml); you don't need to fully understand what's in that file, however the gist is that we have several parallel checks (unit tests, gpu tests) on several different setups (OS X, various PyTorch releases) that covers nearly all of the environments in which we expect ParlAI to function.

:::{note}
Changes to anything within the [`parlai/tasks`](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks) directory will trigger a failing test; a bot on GitHub will comment on your Pull Request with a note on what test to run manually.
:::

### Types of Tests in ParlAI

There are several types of tests in ParlAI, and it is important to understand the utility of each:

#### Unit Tests

Unit tests are tests that measure basic functionality of core ParlAI constructs. Breaking a unit test implies that something fundamental about the pull request is flawed, or at least breaks some well-defined ParlAI paradigm. As such, ParlAI has unit tests for *several* commonly used scripts and abstractions, including [model training](https://github.com/facebookresearch/ParlAI/blob/master/tests/test_train_model.py), [argument parsing](https://github.com/facebookresearch/ParlAI/blob/master/tests/test_params.py), [metrics tracking](https://github.com/facebookresearch/ParlAI/blob/master/tests/test_metrics.py), and [higher level agent abstractions](https://github.com/facebookresearch/ParlAI/blob/master/tests/test_torch_agent.py). These tests generally live in the top level [`tests`](https://github.com/facebookresearch/ParlAI/tree/master/tests) directory in ParlAI.
#### Data Tests
There is one main test that is required to pass when one adds tasks and datasets to ParlAI, and GitHub will comment on your pull request to run [the test](https://github.com/facebookresearch/ParlAI/tree/master/tests/datatests/test_new_tasks.py) manually if you add a new task.
#### Task Tests
The data test is a barebones test that simply verifies the teacher data is correctly formatted in appropriate ParlAI [`Message`](parlai.core.message.Message) dicts. However, it is sometimes good to write more nuanced, specific tests for commonly used datasets, to ensure that any changes or updates to the teacher do not mar the specific expected teacher output. These tests are found in the [`tests/tasks`](https://github.com/facebookresearch/ParlAI/tree/master/tests/tasks) directory and include tests for the [Wizard of Wikipedia](https://github.com/facebookresearch/ParlAI/blob/master/tests/tasks/test_wizard_of_wikipedia.py) and [Blended Skill Talk](https://github.com/facebookresearch/ParlAI/blob/master/tests/tasks/test_blended_skill_talk.py) tasks, among others.
#### Nightly Tests
Nightly tests are tests that require significantly more computational resources than your standard unit test; the majority of nightly tests are [GPU tests](https://github.com/facebookresearch/ParlAI/tree/master/tests/nightly/gpu), i.e., the tests require GPUs to run. Nightly tests are commonly used to test models in the ParlAI model zoo, ensuring that the models either train correctly or perform appropriately on their respective tasks. These tests are important as they prevent code changes from introducing regressions in model performance for known, pre-trained models.

### Writing a Test

Writing and running tests in ParlAI is **not** a gargantuan task - we offer several utility functions that hopefully make your life incredibly easy! Seriously! I will first enumerate a few of these functions below, and then walk through writing of both a **Unit Test**.

#### Common Testing Utilities

We offer several testing utility functions in our [`testing` utilities](https://github.com/facebookresearch/ParlAI/blob/master/parlai/utils/testing.py) file. Below are some commonly used, and very helpful, abstractions:

##### **Test Decorators**

###### skipIf, skipUnless

We offer some decorators which are useful for marking tests to run only in certain conditions. These these are the [`skipIf`](parlai.utils.testing.skipIfGPU) or [`skipUnless`](parlai.utils.testing.skipUnlessTorch) functions in the file. E.g., suppose you wanted to only have a test run when there was access to a GPU; you might have the following code in your test file:

```python
import parlai.utils.testing as testing_utils
import unittest

class MyTestClass(unittest.TestCase):

    @testing_utils.skipUnlessGPU
    def test_gpu_functionality(self):
        ...
```

###### retry

Or, if your test is "flaky" (sometimes the test fails but the majority of the time it passes), you might want to decorate with a [**`retry`**](parlai.utils.testing.retry):

```python
import parlai.utils.testing as testing_utils
import unittest

class MyTestClass(unittest.TestCase):

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_that_is_flaky(self):
        ...
```

##### Context Managers

We offer a few context managers for [capturing output](parlai.utils.testing.capture_output), [managing temporary directories](parlai.utils.testing.tempdir), and [raising timeouts](parlai.utils.testing.timeout). These can be used in the following ways:

```python
import parlai.utils.testing as testing_utils
import unittest

class MyTestClass(unittest.TestCase):

    def test_that_spews_output(self):
        with testing_utils.capture_output() as output:
            # do something that prints to stdout
        # do something with the captured output

    def test_with_tempdir(self):
        with testing_utils.tempdir() as tmpdir:
            print(tmpdir)  # prints a folder like /tmp/randomname

    def test_with_timeout(self):
        with testing_utils.timeout(time=30):
            # Run something that you want to take less than 30 seconds
```

##### Running Common ParlAI Scripts

Finally, the testing utils offer really easy ways to both [train](parlai.utils.testing.train_model) and [evaluate](parlai.utils.testing.eval_model) models, and also display [teacher](parlai.utils.testing.display_data) and [model](parlai.utils.testing.display_model) outputs. These functions will automatically generate model files if necessary, and will return the appropriate output (or valid/test performance, where applicable). All you need to do is pass a dict of relevant arguments.

```python
import parlai.utils.testing as testing_utils
import unittest

class MyTestClass(unittest.TestCase):

    def test_train_model(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests',
                model='examples/tra',
                num_epochs=10,
                batchsize=16,
            )
        )
        self.assertGreater(valid['accuracy'], 0.8)
        ...

    def test_eval_model(self):
        opt = dict(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )

        valid, test = testing_utils.eval_model(opt)

        self.assertEqual(valid['accuracy'], 1)
        ...

    # etc...
```

#### Integration Testing Teachers

As you may have seen in some of the examples above, we often do not use real ParlAI tasks for testing, as they are generally too large (and require downloads), whereas simple integration task teachers provide easy ways to quickly measure model performance. The integration testing teachers can be used via specifying `task='integration_tests:TestTeacher`, and they are all found in [`parlai/tasks/integration_tests/agents.py`](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/integration_tests/agents.py). The default teacher is a teacher whose text and label are identical, and provides some label candidates as well. Other variations include a `ReverseTeacher` (the label is the reverse of the text), an `ImageTeacher` (which provides dummy images for testing image-based models), `MultiturnTeacher` (for testing agents on multi-turn tasks), etc.


#### Writing your own **Unit Test**

Now that we've enumerated what's available to you, let's right a unit test! Woo! I can hear your excitement from here! To make this as useful as possible, we will walk through a simple change recently (as of this writing) made in ParlAI that required a thoughtful unit test (full source code is [here](https://github.com/facebookresearch/ParlAI/blob/master/tests/test_tga.py)).

##### Scenario
For our generative models, we provide an inference interface for various types of generation: nucleus sampling, top-k sampling, beam search, etc. We additionally provide finer-grained options, such as blocking from generating repeated n-grams in the current sentence, or even n-grams in the recent dialogue history. The `--beam-context-block-ngram` controls this functionality, and if set to a positive nonzero number N, the [`TorchGeneratorAgent`](parlai.core.torch_generator_agent.TorchGeneratorAgent) will ensure that there are no repeated N-grams in generation that appeared in **any part** of the dialogue history.

One important caveat to this parameter is that the `context` was recently defined as anything in the model's tracked history. This is unfortunately controlled by truncation parameters, and functions in such a way that if the dialogue proceeds long enough, the model does not know what was said in the distant history - in this situation, it would be possible for a model to repeat an N-gram from the beginning of the conversation. To remedy this, we decided to add a new command line argument `--beam-block-full-context`, that, if `True`, will allow the [`TreeSearch`](parlai.core.torch_generator_agent.TreeSearch) objects to access the **full** dialogue history, regardless of model truncation parameters.

##### Writing the Test

On the surface, this is a rather simple change - the context for beam search is now the whole dialogue history, not just the model's truncated history. Thus, to test that this change works, we would like to investigate what is given as `context` for the [`TreeSearch`](parlai.core.torch_generator_agent.TreeSearch) algorithms, and whether it indeed is the full dialogue history.

First, we'll set up the test class and our test method, `test_full_context_block`, and create our agent.

```python
import unittest
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils

class TestTreeSearch(unittest.TestCase):

    def test_full_context_block(self):
        args = [
            '--model-file', 'zoo:unittest/transformer_generator2/model',
            '--inference', 'beam',
            '--truncate', '1024',
        ]
        pp = ParlaiParser(True, True)
        agent = create_agent(pp.parse_args(args), require_model_exists=True)

```

There are two important things to note here:

1. The agent has a truncation of `1024`, meaning that any tokens beyond this number are not seen by the agent as dialogue history.
2. For this agent, we **have not** specified `--beam-block-full-context True`. We want to first test that our assumption about what occurs **prior to this change** is true, and then test that our new parameter works as intended.

To test that indeed the context is truncated, we'll have our agent observe two dialogue turns, each of length 1024 (for a total length of 2048), and assert that the context from `_get_context`, which is what is provided to the [`TreeSearch`](parlai.core.torch_generator_agent.TreeSearch) object, is only 1024 tokens (note that the agent's dictionary is such that the string '1 2 3 4' is parsed into tokens `[5, 4, 6, 7]`).

```python
...

    def test_full_context_block(self):
        ...
        obs = {'text': '1 2 3 4 ' * 256, 'episode_done': False}
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is the same (truncation)
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)
```

Now that we've tested our old functionality, let's see what happens when we add our new argument:

```python
...

    def test_full_context_block(self):
        ...
        # previous tests
        ...
        args += ['--beam-block-full-context', 'true']
        agent2 = create_agent(pp.parse_args(args), True)
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(agent2._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is larger now
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(
            agent2._get_context(batch, 0).tolist(),
            [5, 4, 6, 7] * 256 + [3] + [5, 4, 6, 7] * 256,
        )  # 3 is end token.

```

We have now verified that the context for the [`TreeSearch`](parlai.core.torch_generator_agent.TreeSearch) algorithms indeed includes longer than 1024 tokens. Success!!

For closure, the full test file will look like this:

```python
import unittest
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils

class TestTreeSearch(unittest.TestCase):

    def test_full_context_block(self):
        args = [
            '--model-file', 'zoo:unittest/transformer_generator2/model',
            '--inference', 'beam',
            '--truncate', '1024',
        ]
        pp = ParlaiParser(True, True)
        agent = create_agent(pp.parse_args(args), True)

        obs = {'text': '1 2 3 4 ' * 256, 'episode_done': False}
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is the same (truncation)
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        args += ['--beam-block-full-context', 'true']
        agent2 = create_agent(pp.parse_args(args), True)
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(agent2._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is larger now
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(
            agent2._get_context(batch, 0).tolist(),
            [5, 4, 6, 7] * 256 + [3] + [5, 4, 6, 7] * 256,
        )  # 3 is end token.
```


### Running Your Test

Now that you've written your test, you can try running it with the following command:

```bash
$ pytest -k TreeSearch
============= test session starts ==============

collected 362 items / 361 deselected / 1 selected

tests/test_tga.py .  [100%]

============= slowest 10 test durations ==============
12.62s call     tests/test_tga.py::TestTreeSearch::test_full_context_block

(0.00 durations hidden.  Use -vv to show these durations.)
============= 1 passed, 361 deselected, 8 warnings in 15.60s =============
```
