import random
SEED = 42


class Blender(object):
    def __init__(self, blend_mode):
        self.blend_mode = blend_mode
        random.seed(SEED)

    def blend(self, dialogs):
        if self.blend_mode == "concat":
            return self._concat(dialogs)
        elif self.blend_mode == "random_interleave":
            return self._random_interleave(dialogs)
        elif self.blend_mode == "fixed_interleave":
            return self._fixed_interleave(dialogs)
        else:
            assert f"Unsupported blend mode : {self.blend_mode}"

    def _concat(self, dialogs):
        concatenated_dialog = []
        for dialog in dialogs:
            concatenated_dialog += dialog
        return concatenated_dialog

    def _random_interleave(self, dialogs):
        pass
        
    def _chunkify(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]  

    def _fixed_interleave(self, dialogs):
        num_tasks = len(dialogs)
        max_dialog_turns = [2]*num_tasks
        max_dialog_turns[0] = 4
        
        # chunkify dialogs into chunks and merge.
        dialog_chunks = [self._chunkify(dialog, max_dialog_turns[task_id]) for task_id, dialog in enumerate(dialogs)]
        num_chunks = [len(chunks) for chunks in dialog_chunks]
        interleaved_dialog = []
        task_id = 0
        while sum(num_chunks) > 0:
            if dialog_chunks[task_id]:
                interleaved_dialog += dialog_chunks[task_id].pop(0)
                num_chunks[task_id] -= 1
            task_id = (task_id + 1) % num_tasks
        return interleaved_dialog
