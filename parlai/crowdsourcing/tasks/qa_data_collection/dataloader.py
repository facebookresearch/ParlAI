from parlai.tasks.squad.agents import DefaultTeacher

class SquadDataLoader(DefaultTeacher):

    def __init__(self, opt):
        super().__init__(opt)

    def act(self):
        data = super().act()
        data.force_set('text', '\n'.join(data['text'].split('\n')[:-1]))
        return data