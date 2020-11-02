import random
SEED = 42


class Blender(object):
    wh_words = ['who', 'where', 'when', 'why', 'what', 'whats', 'how', 'which']

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

    def _can_interleave(self, turn):
        sys_turn = turn["labels"][0]
        return "?" not in sys_turn

    def _chunk_dialogs(self, dialogs, max_chunk_size=2):
        chunks = []
        chunk = []
        for turn in dialogs:
            print(len(dialogs), len(chunks), len(chunk), self._can_interleave(turn))
            if len(chunk) >= max_chunk_size and self._can_interleave(turn):
                chunks.append(chunk)
                chunk = []
            chunk.append(turn)
        chunks.append(chunk)
        return chunks

    def _random_interleave(self, dialogs):
        pass
        
    def _fixed_interleave(self, dialogs):
        num_tasks = len(dialogs)
        max_dialog_turns = [2]*num_tasks
        
        # chunkify dialogs into chunks and merge.
        dialog_chunks = [self._chunk_dialogs(dialog, max_dialog_turns[task_id]) for task_id, dialog in enumerate(dialogs)]
        num_chunks = [len(chunks) for chunks in dialog_chunks]
        interleaved_dialog = []
        task_id = 0
        while sum(num_chunks) > 0:
            if dialog_chunks[task_id]:
                interleaved_dialog += dialog_chunks[task_id].pop(0)
                num_chunks[task_id] -= 1
            task_id = (task_id + 1) % num_tasks
        return interleaved_dialog
