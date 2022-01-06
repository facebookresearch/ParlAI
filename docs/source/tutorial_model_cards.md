# Generating Model Cards

**Author**: Wendy Zhang


## What is a model card?

Think of model cards as a condensed medical card for models. It is a great way for people who might not have the time to read a paper in detail to get the gist of what a model is doing, the datasets involved, how it is performing, and any concerns that the author might have about the model.

You can check out the [Model Cards for Model Reporting paper](https://arxiv.org/pdf/1810.03993.pdf), and here's a sample model card for the [Blenderbot2.0 2.7B model](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/blenderbot2/model_card.md). In addition, here is a [link](https://github.com/facebookresearch/ParlAI/tree/main/docs/sample_model_cards) to some more model card examples. 

## The Process
There are two steps in generating the model cards.
![imageonline-co-whitebackgroundremoved (3)](https://user-images.githubusercontent.com/14303605/128065136-9403281c-3124-488e-be1d-81b9262b7758.png)

For both steps, we should specify the following arguments:
- `--model-file / -mf`: the model file
- `--folder-to-save / -fts`: the location where we're saving reports

and add the command `--mode gen` to signify we're in (report) generation mode.

### Step 1: Generating reports
In general, we can use a command like this for report generation:
```
# template
parlai gmc -mf <model file> -fts <folder name> --mode gen
# sample
parlai gmc -mf zoo:dialogue_safety/multi_turn/model -fts safety_single --mode gen
```

However, depending on the situation, we might need to add these arguments as well:
- `--wrapper / -w` **only if** the model is a generation model
   - check the [safety bench](https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_bench) for more info
- `--model-type / -mt` **only if** the model isn't added to or already in [`model_list.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/model_list.py)
   - possible choices include `ranker`, `generator`, `classifier`, `retriever`
- `--task / -t` and `--evaltask/-et` **only if** the original model.opt used task/datasets not in the form of a teacher or if the task/dataset is no longer accessible
   - tasks starting with `fromfile` or `jsonfile` will be ignored unless `--ignore-unfound-tasks` is set to False (by default, it's true)


In addition, if the model itself needs certain arguments (ie. `--search-server`), we should specify them at this stage too. We can also add `--batchsize` for faster generation.

Check out the section about [generating reports](#details-of-report-generation) for more information on the report generation process and how to generate single reports (very useful for debugging).

### Step 2: Model Card Generation
If some kind of model description has already been added to the [model_list.py](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/model_list.py) (distinguished by `path`, which should be the same as `model_file`), and reports were sucessfully generated in the step before, then we can simply run the following command 
   ```
   # template
   parlai gmc -mf <model file> -fts <folder to save>
   # example
   parlai gmc -mf zoo:dialogue_safety/multi_turn/model -fts safety_multi
   ```

### Examples 
Here are some samples commands: 
- Dialogue Safety (multi-turn)
```
parlai gmc -mf zoo:dialogue_safety/multi_turn/model -fts safety_multi -bs 128  --mode gen -t dialogue_safety:wikiToxicComments,dialogue_safety:adversarial:round-only=False:round=1,dialogue_safety:multiturn -et dialogue_safety:wikiToxicComments,dialogue_safety:adversarial:round-only=False:round=1,dialogue_safety:multiturn --data-parallel False
parlai gmc -mf zoo:dialogue_safety/multi_turn/model -fts safety_multi
```
- Blenderbot 90M
```
parlai gmc -mf zoo:blender/blender_90M/model -fts blenderbot_90M -w blenderbot_90M -bs 128 --mode gen
parlai gmc -mf zoo:blender/blender_90M/model -fts blenderbot_90M
```


##  Report Generation Details
![imageonline-co-whitebackgroundremoved (4)](https://user-images.githubusercontent.com/14303605/128233882-4c77770d-9703-466f-b1a2-7f2395c5c2f6.png) 

In the end, it should generate the following reports under the `--folder-to-save`
- a folder `data_stats/` that contains the data stats of the training set
- a `eval_results.json` that contains the evaluation results based on the evaltasks
- a `sample.json` file contain a sample input and output from the model
- for generators, it should generate a folder `safety_bench_res` that contains the safety_bench results ([click here to learn more about the safety bench](https://github.com/facebookresearch/ParlAI/tree/main/projects/safety_bench)).


Here are some images of the expected behavior:

- *Successful generations should end with a green message like this:*
<img width="679" alt="Screen Shot 2021-07-26 at 3 58 33 PM" src="https://user-images.githubusercontent.com/14303605/127069754-b99cec8c-6fac-4d32-bbca-f4972f6c5b5e.png">

- *Unsucessful generations should tell us which reports are missing and why.*
<img width="1790" alt="Screen Shot 2021-07-26 at 11 32 17 AM" src="https://user-images.githubusercontent.com/14303605/127040345-e8ec6afa-60da-484e-8e68-955f592cec8b.png">

- *When tasks are dropped due to being unaccessible or in a `fromfile` or `jsonfile` format, it should look like this (w/o the blackout)*
<img width="581" alt="Screen Shot 2021-08-06 at 9 10 56 AM" src="https://user-images.githubusercontent.com/14303605/128540191-df949a10-3aba-48e3-a601-a6b97b1dca36.png">


### Generating single reports
Sometimes, you might want to generate only certain reports. In this case, instead of using `--mode gen`, we should use following possibilites:
- `--mode gen:data_stats` to generate the `data_stats/` folder
- `--mode gen:eval` to generate the `eval_results.json` file (evaluation results)
- `--mode gen:safety` to generate the  `safety_bench_res` folder 
- `--mode gen:sample` to generate the `sample.json` file

## Optional Customizations

- Use `--evaluation-report-file` to specify the location of your own evaluation report file.
- Use `--mode editing/final` to specify which mode you would like to use for model card generation.
   
   Currently, there are two different modes `editing` or `final` for step 2.
   For the `editing` mode, the code will generate messages like  this:

   > :warning: missing *section name*: Probably need to be grabbed from paper & added to model_list.py by u (the creator) :warning:

   In `final` mode, such messages will not exist. By default, the `mode` is `editing`. 



### Using `--extra-args-path`

We can use `--extra-args-path` to pass in longer arguments. By default, the `---extra-args-path` will be `<folder-to-save>/args.json`, so if we create a file at that location, we don't need to add `args.json`.

#### Adding Custom Dataset and Model Info
By default, the code will try to find a sections in [`model_list.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/model_list.py). However, instead of changing `model_list.py`, we can also pass in a `.json` file to `--extra-args-path` with out new section. Here's us trying to add the intended use section

```
# args.json
{
   "extra_models": {
      "zoo:blender/blender_90M/model": {
         # section name (lowercased and underscores removed): section content
         "privacy": "Our model is intended for research purposes only, and is not yet production ready...."
      }
   }   
}
```

Similarly, if we don't want to touch [`task_list.py`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/task_list.py) (information about the tasks), we can also pass the details via `--extra-args-path`. Here's us trying add a description for `dummy_task`:
```
# args.json
{
   "extra_tasks": {
      "dummy_task": {
         # type of info: info
         "description": "This is a dummy task, not a real task"
      }
   }   
}
```
The information passed via this method can partially overwrite what's written in `task_list.py` and `model_list.py`. 


#### Add Custom Sections or Changing Section Order

To add sections, there's two ways to do this. 

1. After we generate the inital model card, we can directly edit the generated markdown file.

3. If there's a lot section movement or deletion, use add a `user_sections` key to specify the entire section order to the `.json` file that we pass to `--extra-args-path`. For instance, this is the default order of sections: 
    ```
         section_list = [
            "model_details",
            "model_details:_quick_usage",
            "model_details:_sample_input_and_output",
            "intended_use",
            "limitations",
            "privacy",
            "datasets_used",
            "evaluation",
            "extra_analysis",
            "related_paper",
            "hyperparameters",
            "feedback",
         ]
    ```
   Note that adding `:_` implies that it's a subsection, and I would advise to use underscore `_` in place of spaces (don't worry; they'll be changed back to spaces for the section title). 

   Here's us trying to to reverse the order and remove the model_details section (for kudos):
   ```
   # args.json
   {
      "user_sections": [
         "feedback",
         "hyperparameters",
         "related_paper",
         "extra_analysis",
         "evaluation",
         "datasets_used",
         "privacy",
         "limitations",
         "intended_use"
      ]
   }   
   ```
