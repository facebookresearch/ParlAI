from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import os
import pdb




class CNNDMTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'sst'

        opt['datafile'] = self._path(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        return os.path.join(opt['datapath'], 'cnn', 'stories')

    def setup_data (self, path):
        print('loading: ' + path)

        def extract_data_and_labels(text):
            text_sections = text.split('@highlight')
            return text_sections[0],text_sections[1:]

        self.question = 'What is the summary?'

        new_episode = True

        for file in os.listdir(path):
            if file.endswith('.story'):
                with open(os.join(path,file)) as file_data:
                    data, label = extract_data_and_labels(file_data.read())

                yield (data + '\n' + self.question, label, None, None), new_episode

class DefaultTeacher(CNNDMTeacher):
    pass
