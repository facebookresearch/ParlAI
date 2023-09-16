# Understanding and adding metrics

Author: Stephen Roller

## Introduction and Standard Metrics

:::{tip} List of metrics
If you're not sure what a metric means, refer to our [List of metrics](#list-of-metrics).
:::

ParlAI contains a number of built-in metrics that are automatically computed when
we train and evaluate models. Some of these metrics are _text generation_ metrics,
which happen any time we generate a text: this includes F1, BLEU and Accuracy.

For example, let's try a Fixed Response model, which always returns a given fixed
response, and evaluate on the DailyDialog dataset:

```
$ parlai eval_model --model fixed_response --task dailydialog --fixed-response "how may i help you ?"
... after a while ...
14:41:40 | Evaluating task dailydialog using datatype valid.
14:41:40 | creating task(s): dailydialog
14:41:41 | Finished evaluating tasks ['dailydialog'] using datatype valid
    accuracy  bleu-4  exs    f1
    .0001239 .002617 8069 .1163
```

We see that we got 0.01239% accuracy, 0.26% BLEU-4 score, and 11.63% F1 across
8069 examples. What do those metrics means?

- Accuracy: this is perfect, exact, matching of the response, averaged across
  all examples in the dataset
- BLEU-4: this is the [BLEU score](https://en.wikipedia.org/wiki/BLEU) between
  the predicted response and the reference response. It is measured on
  tokenized text, and uses NLTK to compute it.
- F1: This is the [Unigram](https://en.wikipedia.org/wiki/N-gram) F1 overlap
  between your text and the reference response.
- exs: the number of examples we have evaluated

If you don't see the BLEU-4 score, you may need to install NLTK with
`pip install nltk`.

We can also measure [ROUGE](https://en.wikipedia.org/wiki/ROUGE_%28metric%29). Note
that we need to `pip install py-rouge` for this functionality:

```
$ parlai eval_model --model fixed_response --task dailydialog --fixed-response "how may i help you ?" --metrics rouge
14:47:24 | creating task(s): dailydialog
14:47:31 | Finished evaluating tasks ['dailydialog'] using datatype valid
    accuracy  exs    f1  rouge_1  rouge_2  rouge_L
    .0001239 8069 .1163   .09887  .007285   .09525
```

One nice thing about metrics is that they are automatically logged to the
`.trainstats` file, and within Tensorboard (when enabled with
`--tensorboard-log true`. As such, metrics are more reliable than adding print
statements into your code.



### Agent-specific metrics

Some agents include their own metrics that are computed for them. For example,
generative models automatically compute `ppl`
([perplexity](https://en.wikipedia.org/wiki/Perplexity)) and `token_acc`, both
which measure the generative model's ability to predict individual tokens.  As
an example, let's evaluate the [BlenderBot](https://parl.ai/projects/recipes/)
90M model on DailyDialog:

```
$ parlai eval_model --task dailydialog --model-file zoo:blender/blender_90M/model --batchsize 32
...
14:54:14 | Evaluating task dailydialog using datatype valid.
14:54:14 | creating task(s): dailydialog
...
15:26:19 | Finished evaluating tasks ['dailydialog'] using datatype valid
    accuracy  bleu-4  ctpb  ctps  exps  exs    f1  gpu_mem  loss      lr  ltpb  ltps   ppl  token_acc  total_train_updates   tpb   tps
           0 .002097 14202 442.5 6.446 8069 .1345    .0384 2.979 7.5e-06  3242   101 19.67      .4133               339012 17445 543.5
```

Here we see a number of extra metrics, each of which we explain below. They may be
roughly divided into diagnostic/performance metrics, and modeling metrics. The
modeling metrics are:

- `ppl` and `token_acc`: the perplexity and per-token accuracy. these are generative
  performance metrics.

The diagnostic metrics are:

- `tpb`, `ctpb`, `ltpb`: stand for tokens per batch, context-tokens per batch,
  and label-tokens per batch. These are useful for measuring how dense the
  batches are, and are helpful when experimenting with [dynamic
  batching](tutorial_fast).  tpb is always the sum of ctpb and lptb.
- `tps`, `ctps`, `ltps`: are similar, but stand for "tokens per second". They
  measure how fast we are training. Similarly, `exps` measures examples per
  second.
- `gpu_mem`: measures _roughly_ how much GPU memory your model is using, but it
  is only approximate. This is useful for determining if you can possibly increase
  the model size or the batch size.
- `loss`: the loss metric
- `total_train_updates`: the number of SGD updates this model was trained for.
  You will see this increase during training, but not during evaluation.

## Adding custom metrics

Of course, you may wish to add your own custom metrics: whether this is because
you are developing a special model, special dataset, or otherwise want other
information accessible to you. Metrics can be computed by either _the teacher_ OR
_the model_. Within the model, they may be computed either _locally_ or _globally_.
There are different reasons for why and where you would want to choose each
location:

- __Teacher metrics__: This is the best spot for computing metrics that depend
  on a specific dataset. These metrics will only be available when evaluating
  on this dataset. They have the advantage of being easy to compute and
  understand. An example of a modeling metric is `slot_p`, which is part of
  some of our Task Oriented Datasets, such as
  [`google_sgd`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/google_sgd/agents.py)
- __Global metrics__ (model metric): Global metrics are computed by the model,
  and are globally tracked. These metrics are easy to understand and track, but
  work poorly when doing multitasking. One example of a global metric includes
  `gpu_mem`, which depends on a system-wide memory usage, and cannot be tied to
  a specific task.
- __Local metrics__ (model metric): Local metrics are the model-analogue of
  teacher metrics.  They are computed and recorded on a per-example basis, and
  so they work well when multitasking. They can be extremely complicated for
  some models, however. An example of a local metric includes perplexity, which
  should be computed on a per-example basis, but must be computed by the model,
  and therefore cannot be a teacher metric.

We will take you through writing each of these methods in turn, and demonstrate
examples of how to add these metrics in your setup.

## Teacher metrics

Teacher metrics are useful for items that depend on a specific dataset.
For example, in some of our task oriented datasets, like
[`google_sgd`](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/google_sgd/agents.py),
we want to additionally compute metrics around slots.

Teacher metrics can be added by adding the following method to your teacher:

```python
    def custom_evaluation(
        self,
        teacher_action: Message,
        labels: Optional[Tuple[str]],
        model_response: Message,
    ) -> None:
        pass
```

The signature for this method is as follows:
- `teacher_action`: this is the last message the teacher sent to the model. This likely
   contains a "text" and "labels" field, as well as any custom fields you might
   have.
- `labels`: The gold label(s). This can also be found as information in the
  `teacher_action`, but it is conveniently extracted for you.
- `model_response`: The full model response, including any extra fields the model
   may have sent.

Let's take an actual example. We will add a custom metric which calculates
how often the __model__ says the word "hello", and call it `hello_avg`.

We will add a [custom teacher](tutorial_task). For this example, we will use
the `@register` syntax you may have seen in our [quickstart
tutorial](tutorial_quick).

```python
from parlai.core.loader import register_teacher
from parlai.core.metrics import AverageMetric
from parlai.tasks.dailydialog.agents import DefaultTeacher as DailyDialogTeacher

@register_teacher("hello_daily")
class CustomDailyDialogTeacher(DailyDialogTeacher):
    def custom_evaluation(
        self, teacher_action, labels, model_response
    ) -> None:
        if 'text' not in model_response:
            # model didn't speak, skip this example
            return
        model_text = model_response['text']
        if 'hello' in model_text:
            # count 1 / 1 messages having "hello"
            self.metrics.add('hello_avg', AverageMetric(1, 1))
        else:
            # count 0 / 1 messages having "hello"
            self.metrics.add('hello_avg', AverageMetric(0, 1))

if __name__ == '__main__':
    from parlai.scripts.eval_model import EvalModel

    EvalModel.main(
       task='hello_daily',
       model_file='zoo:blender/blender_90M/model',
       batchsize=32,
    )
```

If we run the script, we will have a new metric in our output:

```
18:07:30 | Finished evaluating tasks ['hello_daily'] using datatype valid
    accuracy  bleu-4  ctpb  ctps  exps  exs    f1  gpu_mem  hello_avg  loss  ltpb  ltps   ppl  token_acc  tpb   tps
           0 .002035  2172   230 3.351 8069 .1346   .05211      .1228 2.979 495.9 52.52 19.67      .4133 2668 282.6
```

__What is AverageMetric?__

Wait, what is this
[AverageMetric](parlai.core.metrics.AverageMetric)? All metrics
you want to create in ParlAI should be a
[Metric](parlai.core.metrics.Metric) object. Metric objects
define a way of instantiating the metric, a way of combining it with a
like-metric, and a way of rendering it as a single float value. For an
AverageMetric, this means we need to define a numerator and a denominator; the
combination of AverageMetrics adds their numerators and denominators
separately. As we do this across all examples, the numerator will be the number
of examples with "hello" in it, and the denominator will be the total number of
examples. When we go to print the metric, the division will be computed at the
last second.

If you're used to writing machine learning code in one-off scripts, you may ask
why do I need to use this metric? Can't I just count and divide myself? While
you can do this, your code could not be run in [_distributed
mode_](tutorial_fast). If we only returned a single float, we would not be able
to know if some distributed workers received more or fewer examples than
others. However, when we explicitly store the numerator and denominator, we can
combine and reduce the across multiple nodes, enabling us to train on hundreds
of GPUs, while still ensuring correctness in all our metrics.

In addition to AverageMetric, there is also
[SumMetric](parlai.core.metrics.SumMetric), which keeps a running
sum. SumMetric and AverageMetric are the most common ways to construct custom
metrics, but others exist as well. For a full list (and views into advanced
cases), please see the [metrics API documentation](core/metrics_api).

## Agent (model) level metrics

In the above example, we worked on a metric defined by a Teacher. However,
sometimes our models will have special metrics that only they want to compute,
which we call an Agent-level metric. Perplexity is one example.

To compute model-level metrics, we can define either a _Global_ metric, or a
_Local metric_. Global metrics can be computed anywhere, and are easy to use,
but cannot distinguish between different teachers when multitasking. We'll look
at another example, counting the number of times the teacher says "hello".

### Global metrics

A global metric is computed anywhere in the model, and has an
interface similar to that of the teacher:

```python
agent.global_metrics.add('my_metric', AverageMetric(1, 2))
```

Global metrics are called as such because they can be called anywhere in agent
code. For example, we can add a metric that counts the number of times the
model sees the word "hello" in `observe`. We'll do this while extending
the `TransformerGeneratorAgent`, so that we can combined it with the BlenderBot
model we used earlier.

```python
from parlai.core.metrics import AverageMetric
from parlai.core.loader import register_agent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


@register_agent('GlobalHelloCounter')
class GlobalHelloCounterAgent(TransformerGeneratorAgent):
    def observe(self, observation):
        retval = super().observe(observation)
        if 'text' in observation:
            text = observation['text']
            self.global_metrics.add(
                'global_hello', AverageMetric(int('hello' in text), 1)
            )
        return retval


if __name__ == '__main__':
    from parlai.scripts.eval_model import EvalModel

    EvalModel.main(
        task='dailydialog',
        model='GlobalHelloCounter',
        model_file='zoo:blender/blender_90M/model',
        batchsize=32,
    )
```

__Note that this is very different than the Teacher metric we implemented in the
first half of the tutorial__. In the teacher metric, we were counting the number
of times the _model_ said hello. Here, we are counting the number of times the
_teacher_ said hello.

:::{admonition,tip} How to determine where to implement your custom metric:

- If you want your metric to be _model_-agnostic, then it should be implemented
  in the Teacher.
- If you want your metric to be _dataset_-agnostic, then it should be
  implemented in the Model agent.
- If you need your metric to be both model and dataset agnostic, then you
  should do it within the Model, using a
  [mixin](https://www.residentmar.io/2019/07/07/python-mixins.html) or abstract
  class.
:::

Running the script, we see that our new metric appears. As discussed above, the
value differs slightly because of the difference in semantics.

```
21:57:50 | Finished evaluating tasks ['dailydialog'] using datatype valid
    accuracy  bleu-4  ctpb  ctps  exps  exs    f1  global_hello  gpu_mem  loss  ltpb  ltps   ppl  token_acc   tpb   tps
           0 .002097 14202 435.1 6.338 8069 .1345      .0009914   .02795 2.979  3242 99.32 19.67      .4133 17445 534.4
```

The global metric works well, but have some drawbacks: if we were to start
training on a multitask datasets, we would not be able to distinguish the
`global_hello` of the two datasets, and we could only compute the micro-average
of the combination of the two. Below is an excerpt from a training log with
the above agents:

```
09:14:52 | time:112s total_exs:90180 epochs:0.41
                clip  ctpb  ctps  exps  exs  global_hello  gnorm  gpu_mem  loss  lr  ltpb  ltps   ppl  token_acc  total_train_updates   tpb   tps   ups
   all             1  9831 66874 841.9 8416        .01081  2.018    .3474 5.078   1  1746 11878 163.9      .2370                  729 11577 78752 6.803
   convai2                             3434        .01081                 5.288                 197.9      .2120
   dailydialog                         4982        .01081                 4.868                   130      .2620
```

Notice how `global_hello` is the same in both, because the model is unable to
distinguish between the two settings. In the next section we'll show how to fix
this with local metrics.

__On placement__: In the example above, we recorded the global metric inside
the `observe` function. However, global metrics can be recorded from anywhere.

### Local metrics

Having observed the limitation of global metrics being unable to distinguish
settings in multitasking, we would like to improve upon this. Let's add a local
metric, which is recorded _per example_. By recording this metric per example,
we can unambiguously identify which metrics came from which dataset, and report
averages correctly.

Local metrics have a limitation: they can only be computed inside the scope of
`batch_act`. This includes common places like `compute_loss` or `generate`,
where we often want to instrument specific behavior.

Let's look at an example. We'll add a metric inside the `batchify` function,
which is called from within `batch_act`, and is used to convert from a list of
[Messages](core/messages) objects to a
[Batch](parlai.core.torch_agent.Batch) object. It is where we do things like
padding, etc. We'll do something slightly different than our previous runs.
In this case, we'll count the number of _tokens_ which are the word "hello".

```python
from parlai.core.metrics import AverageMetric
from parlai.core.loader import register_agent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


@register_agent('LocalHelloCounter')
class LocalHelloCounterAgent(TransformerGeneratorAgent):
    def batchify(self, observations):
        batch = super().batchify(observations)
        if hasattr(batch, 'text_vec'):
            num_hello = ["hello" in o['text'] for o in observations]
            self.record_local_metric(
                'local_hello',
                AverageMetric.many(num_hello),
            )
        return batch


if __name__ == '__main__':
    from parlai.scripts.train_model import TrainModel

    TrainModel.main(
        task='dailydialog,convai2',
        model='LocalHelloCounter',
        dict_file='zoo:blender/blender_90M/model.dict',
        batchsize=32,
    )
```

When we run this training script, we get one such output:
```
09:49:00 | time:101s total_exs:56160 epochs:0.26
                clip  ctpb  ctps  exps  exs  gnorm  gpu_mem  local_hello  loss  lr  ltpb  ltps   ppl  token_acc  total_train_updates  tpb   tps  ups
   all             1  3676 63204 550.2 5504  2.146    .1512       .01423 4.623   1 436.2  7500 101.8      .2757                 1755 4112 70704 17.2
   convai2                             3652                       .02793 4.659                 105.5      .2651
   dailydialog                         1852                       .00054 4.587                 98.17      .2863
```

Notice how the `local_hello` metric can now distinguish between hellos coming from
convai2 and those coming from daily dialog? The average hides the fact that one
dataset has many hellos, and the other does not.

Local metrics are primarily worth the implementation when you care about the
fidelity of _train time metrics_. During evaluation time, we evaluate each
dataset individually, so we can ensure global metrics are not mixed up.

__Under the hood__: Local metrics work by including a "metrics" field in the
return message. This is a dictionary which maps field name to a metric value.
When the teacher receives the response from the model, it utilizes the metrics
field to update counters on its side.

(list-of-metrics)=
## List of Metrics
Below is a list of metrics and a brief explanation of each.

:::{note} List of metrics
If you find a metric not listed here,
please [file an issue on GitHub](https://github.com/facebookresearch/ParlAI/issues/new?assignees=&labels=Docs,Metrics&template=other.md).
:::

```{include} metric_list.inc
```
