from typing import Optional
from parlai.core.torch_agent import Batch
from projects.seeker.agents.seeker import ComboFidGoldDocumentAgent

import torch


class ComboFidGoldDocumentAgentWithPrefixTokens(ComboFidGoldDocumentAgent):
    def get_prefix_tokens(self, batch: Batch) -> Optional[torch.LongTensor]:
        # Set the prompts themselves (Message.text) as the prefixes
        # so that the generation is a strict continuation. batch.text_vec
        # contains eos tokens so this copies everything up to
        # but not including them.
        new_vectors = []
        for vector in batch.text_vec.tolist():
            new_vector = []
            for i in vector:
                if i == self.START_IDX:
                    continue
                if i == self.END_IDX:
                    break
                new_vector.append(int(i))
            new_vectors.append(new_vector)

        if batch.batchsize > 1:
            tensor, _ = self._pad_tensor(new_vectors)
        else:
            tensor = torch.LongTensor(new_vectors)

        dev = batch.text_vec.device
        return tensor.to(dev)
