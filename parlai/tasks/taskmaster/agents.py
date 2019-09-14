from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os, json, copy

class TaskMasterTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt)

        if shared and 'convos' in shared:
            # another instance was set up already, just reference its data
            self.convos = shared['convos']
        else:
            # need to set up data from scratch
            data_path = _path(opt)
            self._setup_data(data_path)

        self.reset()

    def _setup_data(self, data_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)

    def num_examples(self):
        return len(self.convos)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        user_q = conversation['utterances'][entry_idx * 2]['text']
        assistant_a = conversation['utterances'][entry_idx * 2 + 1]['text']
        ep_done = False

        if entry_idx * 2 + 1 == conv_len - 1:
            ep_done = True
        action = {
            'id': self.id,
            'text': user_q,
            'episode_done': ep_done
        }
        action['labels'] = [assistant_a]

        return action

class SelfDialogueTeacher(TaskMasterTeacher):
    def __init__(self, opt, shared=None):
        opt['fn'] = "self-dialogs.json"
        super().__init__(opt, shared)

class WozDialogueTeacher(TaskMasterTeacher):
    def __init__(self, opt, shared=None):
        opt['fn'] = "woz-dialogs.json"
        super().__init__(opt, shared)

class SelfDialogueSegmentTeacher(SelfDialogueTeacher):
    def get(self, episode_idx, entry_idx):
        conversation = self.convos[episode_idx]
        conv_len = len(conversation['utterances'])
        # print(conv_len, entry_idx, conversation)
        utterance = conversation['utterances'][entry_idx]['text']
        ep_done = False

        if entry_idx == conv_len - 1:
            ep_done = True
        action = {
            'id': self.id,
            'text': utterance,
            'episode_done': ep_done
        }

        return action

    def _setup_data(self, data_path):
        super()._setup_data(data_path)
        convos_update = []
        for convo in self.convos:
            x = copy.deepcopy(convo['utterances'])
            y = []
            for i in range(0, len(x)):
                if "segments" in x[i]:
                    y += [x[i]]
            convo['utterances'] = y
            if convo['utterances']:
                convos_update += [convo]
        self.convos = convos_update

# Utils
def _path(opt):
    # ensure data is built
    build(opt)
    return os.path.join(opt['datapath'], 'taskmaster-1', opt['fn'])

class DefaultTeacher(TaskMasterTeacher):
    pass
