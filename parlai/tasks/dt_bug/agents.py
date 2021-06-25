from parlai.core.message import Message
from parlai.core.teachers import DialogTeacher


class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.mock_data = [[1, 2, 3], [1, 2, 3, 4]]
        opt['datafile'] = "df"
        super().__init__(opt, shared=shared)

    def setup_data(self, datafile):
        for episode in self.mock_data:
            for ex in episode:
                done = ex == episode[-1]
                yield Message({"text": f"text_{ex}", "labels": [f"label {ex}"]}), done
