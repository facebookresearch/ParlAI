import numpy as np
import re
import torch
from typing import Any, Dict, List, Tuple
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai_internal.tasks.empathy_labeled_utterances.agents import (
    PrevCurrUttEmpathyAgent,
    PrevCurrFutureUttEmpathyAgent,
    FUTURE_SEP_TOKEN,
)

RESPONSE_CLASSIFIER_PATH = "/checkpoint/ccross/trained-classifiers/parlai/s2020_04_01__style_classifier_retrieve_st/model"

# adapted from tone classifier code of https://github.com/fairinternal/ParlAI-Internal/blob/master/projects/style_gen/auto_evals/s2020_04_30__auto_metrics/measure_style_accuracy.py#L58
class BatchClassifier:
    """
    slightly modification to projects.stylegen.model.utils.
    BatchClassifier where we're also passing in the class file listing all styles
    """

    def __init__(self, classifier_path: str, classes_file: str, batch_size: int):
        args_ = f"""--classes-from-file {classes_file} \
                    --model-file {classifier_path} \
                    --fp16-impl mem_efficient \
                    --fp16 True \
                    --model-parallel True \
                    --model parlai_internal.projects.empathy_generation.style_classifier.from_pretrained_model:FromPreTrainedClassifierAgent \
                    --print-scores True \
            """.split()
        parser_ = ParlaiParser(add_model_args=True)
        self.opt = parser_.parse_args(args_)

        self.classifier_cls_agent = create_agent(self.opt)
        self.classifier_cls_agent.model.eval()
        shared = self.classifier_cls_agent.share()
        self.classifier_copies = [
            create_agent_from_shared(shared) for _ in range(batch_size)
        ]
        self.class_list = self.classifier_cls_agent.class_list

    @torch.no_grad()
    def batch_classify(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        all_obs, ground_truths, predictions = [], [], []
        class_list = self.classifier_cls_agent.class_list

        for c, example in zip(self.classifier_copies[: len(batch)], batch):
            c.reset()
            example["episode_done"] = True
            # example["eval_labels"] = [example["style"]]
            ground_truths.append(example["style"])
            obs = c.observe(example)
            all_obs.append(obs)
        try:

            all_acts = self.classifier_cls_agent.batch_act(all_obs)
            for i, (obs, act) in enumerate(zip(all_obs, all_acts)):
                text_fields = re.fullmatch(
                    r"Predicted class: (.+)\nwith probability: ([\d.]+)", act["text"]
                )
                pred_idx = np.argmax(act["probs"])
                predictions.append(class_list[pred_idx])

        except RuntimeError as err:
            raise Exception("Halting due to error in batch.")

        assert len(all_obs) == len(ground_truths) == len(predictions)
        return ground_truths, predictions

    @torch.no_grad()
    def batch_classify_and_rerank(self, batch):
        # all examples in batch should be from same input to generator
        all_obs, all_probs, ground_truths = [], [], []
        class_list = self.classifier_cls_agent.class_list

        for c, example in zip(self.classifier_copies, batch):
            c.reset()
            example["episode_done"] = True
            example["eval_labels"] = [example["style"]]
            ground_truths.append(example["style"])
            obs = c.observe(example)
            all_obs.append(obs)

        try:
            all_acts = self.classifier_cls_agent.batch_act(all_obs)
            for obs, act, gt in zip(all_obs, all_acts, ground_truths):
                class_idx = class_list.index(obs["style"])
                prob = act["probs"][class_idx]
                all_probs.append(prob)

        except RuntimeError as err:
            raise Exception("Halting due to error in batch.")

        sorted_probs, sorted_indices = torch.sort(
            torch.tensor(all_probs), descending=True
        )
        return sorted_probs, sorted_indices


def load_classifier_agent(datatype: str, data_prefix: str, task_name: str):
    if task_name == "prev_curr":
        task = "parlai_internal.tasks.empathy_labeled_utterances.agents:PrevCurrUttEmpathyAgent"
    elif task_name == "prev_curr_future":
        task = "parlai_internal.tasks.empathy_labeled_utterances.agents:PrevCurrFutureUttEmpathyAgent"
    else:
        raise Exception(f"Unknown task: {task_name}")
    task_args = f"--datatype {datatype} \
                    --data-prefix {data_prefix} \
                    --task {task}".split()
    parser_ = ParlaiParser(add_parlai_args=True)
    opt = parser_.parse_args(task_args)
    agent = PrevCurrFutureUttEmpathyAgent(opt)
    return agent
