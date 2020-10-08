import more_itertools
from datasets import load_dataset

from parlai.core.teachers import DialogTeacher


class WritingPromptsDialogTeacher(DialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('WritingPrompts Teacher Args')
        parser.add_argument(
            '--writing-prompts-config-name',
            type='str',
            default="writing_prompts_sentence",
            help="The WritingPrompts huggingface configs name to load.",
        )

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']

        if opt['datatype'].startswith('train'):
            suffix = 'train'
        elif opt['datatype'].startswith('valid'):
            suffix = 'valid'
        else:
            suffix = 'test'

        self.id = 'writing_prompts'

        self.dataset = load_dataset('./huggingface_dataset.py', name=opt["writing_prompts_config_name"], split=suffix)

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)

        for story in self.dataset:

            passage_pairs = more_itertools.chunked(story["passages"], n=2)
            passage_pairs = [p for p in passage_pairs if p[1] is not None]
            passage_pairs_len = len(passage_pairs)
            for i, passage_pair in enumerate(passage_pairs_len):
                end_of_episode = i + 1 == passage_pairs_len
                yield {"text": passage_pair[0]["text"], "labels": passage_pair[1]["text"]}, end_of_episode


class ExtrasTeacher(WritingPromptsDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared, extras=True)


class DefaultTeacher(WritingPromptsDialogTeacher):
    pass
