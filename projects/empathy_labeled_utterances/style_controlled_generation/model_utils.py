import math
import numpy as np
import re
import torch
from typing import Any, Dict, List, Tuple
import warnings
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai_internal.projects.empathy_generation.style_classifier.model_utils import (
    BatchClassifier,
)
from parlai_internal.tasks.empathy_labeled_utterances.agents import (
    ContextAndStyleAgent,
    FUTURE_SEP_TOKEN,
)

HISTORY_DELIMITER = "\n"


class FutureGenerator:
    def __init__(
        self, batch_size: int, beam_size: int = 10
    ):  # batch size here should correspond to main generator's beam size
        print("\nLoading future generator.")
        args_ = f"""-mf zoo:blender/blender_90M/model \
                --fp16 True \
                --fp16-impl mem_efficient \
                --optimizer mem_eff_adam \
                --skip-generation False \
                --model-parallel True \
                --inference beam \
                --beam-size {beam_size}
                """.split()
        parser_ = ParlaiParser(add_model_args=True)
        self.future_gen_opt = parser_.parse_args(args_)
        self.future_generator = create_agent(self.future_gen_opt)
        self.future_generator.model.eval()
        shared = self.future_generator.share()
        self.generator_copies = [
            create_agent_from_shared(shared) for _ in range(batch_size)
        ]

    @torch.no_grad()
    def batch_generate(self, batch):
        observations, generations = [], []
        for c, example in zip(self.generator_copies[: len(batch)], batch):
            c.reset()
            new_example = Message({"text": example})
            if "retriever_reply" in example:
                new_example.force_set("retriever_reply", example["retriever_reply"])
            obs = c.observe(new_example)
            observations.append(obs)
        try:
            all_acts = self.future_generator.batch_act(observations)
            for act in all_acts:
                generations.append(act["text"])  # just keep best respond from beam
        except RuntimeError as err:
            raise Exception("Halting future generator due to error in batch.")
        assert len(batch) == len(
            generations
        ), f"lens {len(observations)} {len(batch)} {len(generations)}"
        return generations


class BatchGeneratorWithFutureClassifier:
    def __init__(
        self,
        generator_path: str,
        classifier_path: str,
        classes_path: str,
        batch_size: int,
        history_size: int,
        style_frac: int,
        beam_size: int = 10,
    ):
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Load generator
        print("\nLoading generator.")
        args_ = f"""--batchsize {batch_size} \
                --history-size {history_size} \
                -mf {generator_path} \
                --skip-generation False \
                --use_style_frac {style_frac} \
                --model-parallel True \
                --inference beam \
                --beam-size {beam_size} \
                --beam-min-length 20 \
                --beam-block-ngram 3 \
                --beam-context-block-ngram 3 \
                --fp16 True \
                --fp16-impl mem_efficient \
                """.split()
        parser_ = ParlaiParser(add_model_args=True)
        self.opt = parser_.parse_args(args_)

        self.generator_cls_agent = create_agent(self.opt)
        self.generator_cls_agent.model.eval()
        shared = self.generator_cls_agent.share()
        self.generator_copies = [
            create_agent_from_shared(shared) for _ in range(batch_size)
        ]

        # load future generator and future-looking classifier
        self.future_generator = FutureGenerator(beam_size)
        self.future_classifier = BatchClassifier(
            classifier_path, classes_path, beam_size
        )

    @torch.no_grad()
    def batch_generate(self, examples: List[Dict[str, Any]]) -> List[str]:
        print(f"{len(examples):d} examples to generate.")
        batched_sample_indices = np.array_split(
            np.arange(len(examples)), math.ceil(len(examples) / self.batch_size)
        )
        print(f"{len(batched_sample_indices):d} batches created.")

        generations = []
        for batch_idx, example_indices in enumerate(batched_sample_indices):
            batch = examples[example_indices[0] : example_indices[-1] + 1]

            all_obs = []
            for c, example in zip(self.generator_copies[: len(batch)], batch):
                c.reset()
                new_example = Message(
                    {
                        "text": example["text"],
                        "personality": example["style"],
                        "episode_done": True,
                    }
                )

                if "retriever_reply" in example:
                    new_example.force_set("retriever_reply", example["retriever_reply"])
                obs = c.observe(new_example)
                # add keys from original example, excluding text)
                example.pop("text")
                obs.update(example)
                all_obs.append(obs)
            try:
                with torch.no_grad():
                    all_acts = self.generator_cls_agent.batch_act(all_obs)
                for obs, act in zip(all_obs, all_acts):
                    # get responses and classify
                    future_generator_input = [
                        obs["text"] + HISTORY_DELIMITER + gen_text
                        for gen_text, gen_prob in act["beam_texts"][: self.beam_size]
                    ]

                    # pass through generator
                    future_generations = self.future_generator.batch_generate(
                        future_generator_input
                    )

                    # finally classify and rerank
                    speaker, listener = obs["speaker_utt"], obs["listener_utt"]
                    classifier_input = []
                    for gen in future_generations:
                        assert gen != "" and gen != " "
                        classifier_input.append(
                            {
                                "text": f"{speaker} {FUTURE_SEP_TOKEN} {gen} {HISTORY_DELIMITER} {listener}",
                                "style": obs["style"],
                            }
                        )

                    (
                        sorted_probs,
                        sorted_indices,
                    ) = self.future_classifier.batch_classify_and_rerank(
                        classifier_input
                    )
                    best_idx = sorted_indices[0].item()  # highest prob match
                    keep_gen = act["beam_texts"][best_idx]
                    obs["model_generation"] = keep_gen
                    obs["prediction_acc"] = (sorted_probs >= 0.5).cpu().tolist()

                    # store by conversation
                    obs.pop("text_vec", None)
                    obs.pop("full_text_vec", None)
                    generations.append(obs)

            except RuntimeError as err:
                warnings.warn(f"Error in batch {batch_idx}! Error:\n\t{err}")
                return generations  # return generations thus far

            print(
                f"Generated roughly {(batch_idx + 1) * self.batch_size:d} " f"examples."
            )
        return generations


def load_gen_agent(datatype: str, data_prefix: str):
    task_args = f"--datatype {datatype} \
                    --data-prefix {data_prefix} \
                    --task parlai_internal.tasks.empathy_labeled_utterances.agents:ContextAndStyleAgent".split()
    parser_ = ParlaiParser(add_parlai_args=True)
    opt = parser_.parse_args(task_args)
    agent = ContextAndStyleAgent(opt)
    return agent
