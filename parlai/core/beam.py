import torch

class Beam(object):
    """
    Beam class serves as a structure to keep beam of hypothesis with corresponding scores, backtracking info etc.
    """
    def __init__(self, beam_size, min_length, padding_token, bos_token, eos_token, cuda):
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.Tensor(self.beam_size).float().zero_()  # recent score for each hypo in the beam
        self.all_scores = []  # self.scores values per each time step
        self.bookkeep = []  # backtracking id to hypothesis at previous time step
        self.outputs = [self.tt.Tensor(self.beam_size).long().fill_(padding_token)]  # output tokens at each time step
        self.eos_top = False
        self.eos_top_ts = None

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def advance(self, softmax_probs):

        voc_size = softmax_probs.size(-1)
        self.all_scores.append(self.scores)
        beam_scores = None

        if len(self.bookkeep) == 0:
            #  the first step
            beam_scores = softmax_probs[0]  # we take only the first hypo into account since all hypos are the same initially
        else:
            #  we need to sum up hypo scores and current softmax scores before topk
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(softmax_probs)
            for i in range(self.outputs[-1].size(0)):
                #  if previous output hypo token had eos - we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    beam_scores[i] = -1e20  # beam_scores[i] is voc_size array for i-th hypo

        flatten_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flatten_beam_scores, self.beam_size, dim=-1)
        # import IPython;
        # IPython.embed()

        self.scores = best_scores
        hyp_id = best_idxs / voc_size
        tok_id = best_idxs % voc_size

        self.outputs.append(tok_id)
        self.bookkeep.append(hyp_id)

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        return self.eos_top

    def get_top_hyp(self):
        top_hyp = []
        endtok = self.outputs[self.eos_top_ts][0].item()
        top_hyp.append(endtok)
        endback = self.bookkeep[-1][0]
        for i in range(self.eos_top_ts-1, -1, -1):
            top_hyp.append(self.outputs[i][endback].item())
            endback = self.bookkeep[i][endback]

        return [i for i in reversed(top_hyp)][1:]



