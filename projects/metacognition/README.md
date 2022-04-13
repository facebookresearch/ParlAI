# Reducing conversational agents' overconfidence through linguistic calibration

## Paper

[Link](https://arxiv.org/abs/2012.14983); accepted to TACL 2022.

## Data

The data, containing fragments of TriviaQA and model responses is not part of the repository, but can be downloaded [here](https://sjmielke.com/data/metacognition_data.zip). Please follow the instructions in that zip file to recreate the data from TriviaQA sources.


## Code

The given code extends the ParlAI system (tested under commit `f9e67b7286fcd7b9adfb94b62c3b5cb8fe0c9038`), running `python setup.py develop` in the `ParlAI/` root.

You'll want to adjust the paths at the top of `parlai/projects/metacognition/wsgi.py` accordingly


### `parlai/projects/metacognition/agents.py`

The file `agents.py` contains three different subclasses:

#### PartialLossExtractingTransformerGeneratorAgent

This agent inherits from `parlai.agents.transformer.generator.GeneratorAgent` and implements only `eval_step`, largely copied from `TorchGeneratorAgent.eval_step`. It extracts losses/perplexities of the generation, the gold, and interesting substrings, and embeddings from a (custom) forward pass, emitting all those into the `.jsonl` (which becomes huge this way)!

#### ControllingTransformerGeneratorAgent

This agent likewise inherits from `parlai.agents.transformer.generator.GeneratorAgent`, but overrides first and foremost `get_temp_history` to insert appropriate control tokens (stochastically with a new CLI parameter) when the observations contain those as a `control_string` key. It is used in conjunction with the following teacher:

#### MCCertaintyControlTeacher

This teacher subclasses `parlai.core.teachers.DialogTeacher` and (hardcodedly) loads MTurk annotations to train a controlled model, supplying the `control_string` key for the controlling agent, adding a CLI parameter to downsize the training set.

#### NoEvidenceUnionControlledTeacher

Used for evaluation, it also subclasses `parlai.core.teachers.DialogTeacher`, but forcibly inserts a desired control string, passed to its constructor, into every data point. It is subclassed by `NoEvidenceUnionForced{IDK,TRY,YEA}Teacher` that set that token.



### Analysis code in a WSGI app

The bulk of my code is all the analysis code I wrote, living in `wsgi.py` alone.

#### Data structures

A run `TriviaQARun` is the output of an `eval_model` call, testing the model on (usually) the TriviaQA dev set (`no-evidence-union`), i.e., getting generations from the model that is to be tested. The Python class also computes some simple regex-based certainty and correctness metrics for each constituent `QASample`, computes overall statistics, and caches all results in a `.pkl` next to the `.jsonl` from which it is instantiated.

Caching is implemented using a base class `CachedGroup` that must be subclassed to be used, implementing at least the second of these methods:
* `loader(f, path)` uses the file object `f` that contains the cache and returns the desired data (default: `return pickle.load(f)`)
* `generator()` is called on a cache miss to generate the desired data
* `dumper(obj, path)` dumps the (created) data to a cache file on disk at `path` (default: `with open(path, "wb") as f: pickle.dump(obj, f, protocol=4)`)


#### What tools are available?

endpoint | use
--- | ---
`/figures/<path:figname>` | statically serve matplotlib figures
`/` or `/summary/<path:folder>` | table of all runs with regex statistics.
`/examples/<int:start>/<int:end>/<path:logfile>` | details of an individual run: predictive n-grams, clustering plots, perplexities, and examples
`/miniannotation/<int:seed>` or `/miniannotation/<int:seed>/<path:annotation_filename>` | pilot annotation GUI for ("trained") authors to annotate a 150 questions sample with some analysis on top
`/pilotconfusion/<path:worker2qp2a_file>` | inter-annotator agreements and confusion matrices from prepared annotation file
`/compare-two-annotations/<string:who1>/<string:who2>` | compare two annotators with `miniannotation`-like GUI
`/predict/<string:metric>/<string:classifier>/<path:logfile>` | preliminary predictions of regex metrics from embeddings using sklearn toolbox
`/mephistojudge_worker/<string:worker_name>` | final GUI for accepting MTurk annotation batches and entire workers with soft-reject and bonuses
`/mephistoresults/<string:task_names>` | looking at individual MTurk batches (w/ accept/reject)
`/complete_mephisto/<path:jsonlpath>`, `/finalmephistoextract`, `/mephistojudge_final` | get MTurk annotations
`/whats_missing/<int:desired_multiplicity>/<int:todo>/<path:jsonlpath>` | see what is missing due to rejected MTurk batches
`/certain_buckets` | bucket MTurk results
`/mturk_pilot.json` | get pilot data from MTurk
`/mturk_majorities.jsonl` | get only data that a majority agrees on from MTurk
`/mturk_binary_majorities/<path:jsonlpath>` | get data where a majority agrees on simplified labels
`/correctness_trace/<string:identifier>`, `/certaintysankey` | Sankey plot to see forcing remove and add knowledge and change certainty, respectively
`/annotatable_data_shuffled_limit_<int:limit>.jsonl` | get a random sample from TriviaQA for MTurk annotation
(`/how_many_regex_correct`, `/predict_certainty`) `/regex_admissibility`, `/bert_admissibility` | compare automatic to human annotation
`/process-sweep-results`, `/join-finetune-sweep-results`, `/compare-finetune-sweep-results/<string:sweepname>`, `/finetune` | code to analyze hyperparameter tuning of controllable models
`/show_miscalibration` | basic table showing miscalibration
`/probe-results/<string:sweepname>/<string:dset>`, `/probe-certainties`, `/probe-ablations`, `/probe-ablations-all` | analyze results of the probabilistic calibrator
`/calibrate` | construct and analyze complete pipeline results
`/vanilla-ngrams` | n-gram analysis


### Mephisto annotation app (`parlai/projects/metacognition/{webapp/,run_mephisto_task.py}`)

Forked from Mephisto's static React example, `run_mephisto_task.py` sets Mephisto parameters, loads the desired data that is to be annotated and starts jobs---it also defines the onboarding example, if there is one.

Most magic then happens in `webapp/src/components/core_components.jsx`, which defines the GUI and annotation logic.

The single most complicated thing in all of this is managing the fact that one HIT should have multiple questions: this works by passing a list of samples to Mephisto and thus the GUI for each HIT with the JS then cycling through that list, refreshing the annotation component with each sample on submission and only when then list is empty sending the collected list of annotations back to Mephisto.


### Other scripts

script | use
--- | ---
`finetune_sweep.py` | contains the parameters for the controllable fine-tuning sweep, using non-public tooling (encapsulated in `run_grid`)
`probe_sweep.py` | same for the calibrator
`bert4calibration_sweep.py` | same for the BERT-based calibrator
`certainty_classify_sweep.py` | same for the BERT-based automatic certainty annotation
`mephistoexport.py` | simplified Mephisto result extraction
`run_forced_evals.bash` | creates runs for analysis for each of the models trained in a sweep


## Data

The most important data is in the folder alongside the code (`parlai/projects/metacognition`), but we also added additional logfiles in `select_data`---as would fit the limit.
