#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate knowledge grounded language modeling data.

Search a provided data source (given via a `--search-server` parameter)
for matching/similar context that may inform someone of how to generate
the next utterance for a given `--task`.

An `OverlapAgent` finds a matching "context" in all of <DATA SOURCE PROVIDED>
that is **distinct** from the provided context.

The goal is to find supporting information that doesn't come directly
from the document / context itself.

Usage:

```shell
python projects/seeker/scripts/generate_lm_data --task convai2 --search-server <relevant_server> \
    --save-dir /path/to/directory/for/log/saving
```

Upon building, one can then do the following to display the data:

```shell
$ parlai dd -t projects.seeker.tasks.lm:ResponseTeacher --root-dir /path/to/directory/for/log/saving/valid_split_0.5_f1_overlap
$ parlai dd -t projects.seeker.tasks.lm:KnowledgeTeacher --root-dir /path/to/directory/for/log/saving/valid_split_0.5_f1_overlap
$ parlai dd -t projects.seeker.tasks.lm:SearchQueryTeacher --root-dir /path/to/directory/for/log/saving/valid_split_0.5_f1_overlap
```
"""
import nltk
import os
import random
import torch
from typing import Optional, List, Tuple
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from parlai.agents.rag.retrievers import RetrieverType, Document
from parlai.agents.fid.fid import SearchQuerySearchEngineFiDAgent
from parlai.agents.tfidf_retriever.utils import filter_word
from parlai.core.agents import Agent, create_agent
from parlai.core.dict import DictionaryAgent
from parlai.core.message import Message
from parlai.core.metrics import F1Metric
from parlai.core.script import ParlaiScript
from parlai.core.worlds import create_task
from parlai.scripts.eval_model import setup_args, get_task_world_logs
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
import parlai.utils.logging as logging
from parlai.utils.misc import TimeLogger
from parlai.utils.world_logging import WorldLogger
from projects.blenderbot2.agents.modules import BB2SearchQuerySearchEngineRetriever

INVALID = '<INVALID>'
DO_NOT_RETRIEVE = '<DO_NOT_RETRIEVE>'


class OverlapAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        SearchQuerySearchEngineFiDAgent.add_cmdline_args(parser, partial_opt)
        group = parser.add_argument_group('CC Overlap Group')
        group.add_argument(
            '--f1-overlap-threshold',
            type=float,
            default=0.5,
            help='Threshold of f1 in order to keep examples',
        )
        group.add_argument(
            '--min-num-search-words',
            type=float,
            default=-1,
            help='Threshold of how many words required in search term to search example (exclusive)',
        )
        group.add_argument(
            '--write-every-n-valid-exs',
            type=float,
            default=1000,
            help='How many valid exs to write for each log',
        )
        parser.set_defaults(rag_retriever_type=RetrieverType.SEARCH_ENGINE.value)
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        if shared:
            self.search_engine = shared['search_engine']
        else:
            self.search_engine = OverlapSearchEngine(
                opt, DictionaryAgent(opt), shared=shared
            )  # type: ignore
        self.threshold = opt['f1_overlap_threshold']
        self.dummy = torch.zeros(1, 1, dtype=torch.long)

    def share(self):
        shared = super().share()
        shared['search_engine'] = self.search_engine
        return shared

    def construct_search_query(self, labels: List[str]) -> List[str]:
        """
        Construct the search query.

        :param observation:
            observation from task

        :return query:
            return search query.
        """
        assert labels
        search_query = [
            ' '.join(
                [
                    w
                    for w in DictionaryAgent.split_tokenize(labels[0])
                    if not filter_word(w)
                ]
            )
        ]
        return search_query

    def get_best_doc(
        self, all_docs: List[Document], labels: List[str]
    ) -> Tuple[Optional[float], Optional[Document], Optional[int]]:
        """
        Given a set of all retrieved docs, determine best fitting document.

        :param all_docs:
            list of all retrieved Documents
        :param labels:
            labels for the current example

        :return (best_f1, best_doc, best_doc_idx):
            return the best document, along with the f1 overlap and index into all_docs
        """
        docs = []
        for i, d in enumerate(all_docs):
            if d.startswith('.'):
                d = d[2:]
            try:
                docs += [(i, s) for s in nltk.sent_tokenize(d)]
            except IndexError:
                # Something's up with the NLTK Sentence tokenizer here.
                docs += [(i, s) for s in d.split('.')]
        f1s, inds = torch.FloatTensor(
            [F1Metric.compute(labels[0], [d]).value() for _, d in docs]
        ).topk(len(docs))
        best_doc = None
        best_doc_idx = None
        best_f1 = None
        for f1, ind in zip(f1s, inds):
            if self.threshold < f1 < 1.0 and labels[0] not in docs[ind][1]:
                best_doc = docs[ind][1]
                best_doc_idx = docs[ind][0]
                best_f1 = f1.item()
                break

        return best_f1, best_doc, best_doc_idx

    def act(self):
        """
        Search for overlap with the observation label.

        Return the best fitting document. A document is valid if the f1 is above the
        threshold AND the f1 is less than 1.0 AND the target label is not in the
        document.
        """
        obs = self.observation
        reply = {'text': INVALID, 'id': self.getID(), 'episode_done': False}
        if obs is None or obs['text'] == DO_NOT_RETRIEVE:
            return Message(reply)

        # construct the search query
        labels = obs.get('labels', obs.get('eval_labels', None))
        search_query = self.construct_search_query(labels)
        if (
            self.opt['min_num_search_words'] > 0
            and len(search_query[0].split()) <= self.opt['min_num_search_words']
        ):
            return Message(reply)

        # retrieve
        self.search_engine.set_search_queries(search_query)
        retrieved, _ = self.search_engine.retrieve_and_score(self.dummy)
        all_docs = [d.get_tokenization_str() for d in retrieved[0]]  # batched

        # Find the right doc
        best_f1, best_doc, best_doc_idx = self.get_best_doc(all_docs, labels)
        if best_doc:
            assert best_doc_idx is not None
            reply['knowledge'] = f'{TOKEN_KNOWLEDGE}{best_doc}{TOKEN_END_KNOWLEDGE}'
            reply['f1_overlap'] = best_f1
            reply['text'] = labels[0]
            reply['retrieved_docs'] = all_docs
            reply['gold_doc'] = all_docs[best_doc_idx]
            reply['search_query'] = search_query[0]
        return Message(reply)


class OverlapSearchEngine(BB2SearchQuerySearchEngineRetriever):
    def pick_chunk(self, query: str, doc_title: str, doc_text: str, doc_url: str):
        return [[doc_text]]


class GenerateLmData(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = setup_args()
        parser.add_argument(
            '--search-server',
            type=str,
            default=None,
            help='search server argument for retrieving documents',
        )
        parser.add_argument(
            '--save-dir', type=str, default=None, help='Path to directory to save data'
        )
        parser.set_params(model='projects.seeker.scripts.generate_lm_data:OverlapAgent')
        return parser

    def run(self):
        datatype = self.opt['datatype'].split(':')[0]
        self.opt['world_logs'] = os.path.join(
            self.opt['save_dir'],
            f"{datatype}_split_{self.opt['f1_overlap_threshold']}_f1_overlap",
        )
        try:
            self.generate_data()
        except:
            logging.error('ERROR')
            self.log()
            import ipdb

            ipdb.set_trace()
            raise

    def log(self):
        """
        Log world logs.
        """
        self.world_logger.reset()  # add final acts to logs
        outfile = self.task_opt['world_logs']
        os.makedirs(outfile, exist_ok=True)
        outfile = f'{outfile}/{self.cnt}_exs_log_part_{self.log_count}.jsonl'
        self.world_logger.write(
            outfile, self.world, file_format=self.opt['save_format']
        )
        self.world_logger._logs = []  # need to manually set this...
        self.log_count += 1

    def run_generation(self):
        """
        Actually run the evaluations.
        """
        # set up logging
        log_every_n_secs = self.opt.get('log_every_n_secs', -1)
        if log_every_n_secs <= 0:
            log_every_n_secs = float('inf')
        log_time = TimeLogger()

        # max number of examples to evaluate
        max_cnt = (
            self.opt['num_examples'] if self.opt['num_examples'] > 0 else float('inf')
        )
        self.cnt = 0
        self.n_valid = 0
        self.log_count = 0
        total_cnt = self.world.num_examples()

        while not self.world.epoch_done() and self.cnt < max_cnt:
            self.cnt += self.opt.get('batchsize', 1)
            self.world.parley()
            acts = self.world.get_acts()
            if acts[-1]['text'] != INVALID:
                try:
                    self.world.acts[0]['text'] += f"\n{acts[-1]['knowledge']}"
                except RuntimeError:
                    self.world.acts[0].force_set(
                        'text', f"{self.world.acts[0]['text']}\n{acts[-1]['knowledge']}"
                    )
                self.world.acts[0]['f1_overlap'] = acts[-1]['f1_overlap']
                self.world_logger.log(self.world)
                self.n_valid += 1
                if (
                    self.n_valid > 0
                    and self.n_valid % self.opt['write_every_n_valid_exs'] == 0
                ):
                    self.log()
            if log_time.time() > log_every_n_secs:
                report = self.world.report()
                report['n_valid'] = self.n_valid
                text, report = log_time.log(
                    report.get('exs', 0), min(max_cnt, total_cnt), report
                )
                logging.info(text)

    def generate_data(self):
        """
        Generate the LM Data.
        """
        random.seed(42)
        # load model and possibly print opt
        agent = create_agent(self.opt, requireModelExists=True)
        agent.opt.log()

        tasks = self.opt['task'].split(',')
        assert len(tasks) == 1
        task = tasks[0]
        logging.info(
            f'Generating data for task {task} using datatype {self.opt.get("datatype")}.'
        )
        logging.warning('Appending `flatten` to mutators.')
        # set up world logger
        self.task_opt = self.opt.copy()  # copy opt since we're editing the task
        self.task_opt['task'] = task
        if not self.task_opt['mutators']:
            self.task_opt['mutators'] = 'flatten'
        else:
            self.task_opt['mutators'] += '+flatten'
        # add task suffix in case of multi-tasking
        self.task_opt['world_logs'] = get_task_world_logs(
            task, self.task_opt['world_logs'], is_multitask=False
        )

        self.world_logger = WorldLogger(self.task_opt)

        self.world = create_task(self.task_opt, agent)  # create worlds for tasks

        self.run_generation()

        # dump world acts to file
        self.log()

        self.world.reset()


if __name__ == '__main__':
    GenerateLmData.main()
