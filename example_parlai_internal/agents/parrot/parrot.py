from parlai.core.torch_agent import TorchAgent, Output

class ParrotAgent(TorchAgent):
    def eval_step(self, batch):
        # for each row in batch, convert tensor to string
        return Output([str(row) for row in batch.text_vec])
