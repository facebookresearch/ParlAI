"""A WritingPrompts dataset file. """

from __future__ import absolute_import, division, print_function

import jsonlines
import more_itertools
from blingfire import text_to_sentences
import os

import datasets

_CITATION = """\
@inproceedings{fan-etal-2018-hierarchical,
    title = "Hierarchical Neural Story Generation",
    author = "Fan, Angela  and
      Lewis, Mike  and
      Dauphin, Yann",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1082",
    doi = "10.18653/v1/P18-1082",
    pages = "889--898",
}
"""

_DESCRIPTION = """\
 The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
 Each story is a creative writing exercise following a prompt.
"""

_URL = "https://github.com/dwlmt/datasets/raw/main/WritingPrompts/WritingPrompts.tar.gz"


class WritingPromptsDatasetConfig(datasets.BuilderConfig):
    """ BuilderConfig for WritingPrompts"""

    def __init__(self, sentence_block_num = 1, **kwargs):
        """
        Args:
            sentence_block_num (int): How many sentences should be in a single block.
            **kwargs: keyword arguments forwarded to super.
        """
        self.sentence_block_num = sentence_block_num
        super(WritingPromptsDatasetConfig, self).__init__(**kwargs)


class WritingPromptsDataset(datasets.GeneratorBasedBuilder):
    """The WritingPrompts dataset is over 300K short stories collected from the reddit forum /r/WritingPrompts/ .
        Each story is a creative writing exercise following a prompt."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = WritingPromptsDatasetConfig
    BUILDER_CONFIGS = [
        WritingPromptsDatasetConfig(name="writing_prompts_sentence", description="Writing Prompts split by sentence."),
        WritingPromptsDatasetConfig(name="writing_prompts_passage", description="Writing Prompts split by passages of 4 sentences.",sentence_block_num=4)
    ]

    def _info(self):

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "passages": [
                        {"text": datasets.Value("string"), "id": datasets.Value("string"), "seq_num": datasets.Value("int32")}
                    ],
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/dwlmt/datasets/tree/main/WritingPrompts",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "WritingPrompts")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "valid.jsonl"),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """

        with jsonlines.open(filepath, mode='r') as reader:

            for obj in reader:

                passages = []

                sentences = text_to_sentences(f"{obj['title']} {obj['text']}").split('\n')
                passages_text = list(more_itertools.chunked(sentences, self.config.sentence_block_num))

                for i, p in enumerate(passages_text):
                    passages.append({"text": " ".join(p), "seq_num": int(i), "id": f"{obj['id']}-{i}"})

                story_dict = {"id": str(obj["id"]), "title": obj["title"], "passages": passages}

                yield str(story_dict["id"]), story_dict
