import re

import torch

from parlai.core.torch_agent import TorchAgent, Output

# Failure modes
ISAID = 1
NOTSENSE = 2
UM = 3
YOUWHAT = 4
WHATYOU = 5
WHATDO = 6

class FeedbackClassifierRegex(object):
    def __init__(self):
        self.failure_regexes = {
            ISAID:      r"i .*(?:said|asked|told).*",
            NOTSENSE:   r"((not|nt|n't).*mak.*sense)|(mak.*no .*sense)",
            UM:         r"u(m|h)+\W",
            YOUWHAT:    r"you.*what\?",
            WHATYOU:    r"what.*you (?:mean|refer|talk).*\?",
            WHATDO:     r"what.*to do with.*\?"
        }
    
    def predict_proba(self, contexts):
        # Do naive for loop for now
        probs = []
        for context in contexts:
            start = context.rindex('__p1__')
            try:
                end = context.index('__null__')
            except ValueError:
                end = len(context)
            last_response = context[start:end]  # includes padding
            failure_mode = self.identify_failure_mode(last_response)
            probs.append(failure_mode is None)
        return torch.FloatTensor(probs)

    def identify_failure_mode(self, text):
        if re.search(self.failure_regexes[ISAID], text, flags=re.I):
            return ISAID
        elif re.search(self.failure_regexes[NOTSENSE], text, flags=re.I):
            return NOTSENSE
        elif re.search(self.failure_regexes[UM], text, flags=re.I):
            return UM
        elif re.search(self.failure_regexes[YOUWHAT], text, flags=re.I):
            return YOUWHAT
        elif re.search(self.failure_regexes[WHATYOU], text, flags=re.I):
            return WHATYOU
        elif re.search(self.failure_regexes[WHATDO], text, flags=re.I):
            return WHATDO
        else:
            return None        

# class FeedbackClassifier(TorchAgent):
#     def __init__(self, opt, shared):
#         super().__init__(opt, shared)
#         self.model = ?

#     def predict_proba(self, observation):
#         raise NotImplementedError

#     def predict(self, observation, threshold=0.5):
#         return self.predict_proba(observation) > threshold

# class FeedbackClassifierModel(FeedbackClassifier):
#     def train_step(self, batch):
#         """Train on a single batch of examples."""
#         if batch.text_vec is None:
#             return
#         batchsize = batch.text_vec.size(0)
#         self.model.train()
#         self.optimizer.zero_grad()

#         cands, cand_vecs, label_inds = self._build_candidates(
#             batch, source=self.opt['candidates'], mode='train')

#         scores = self.score_candidates(batch, cand_vecs)
#         _, ranks = scores.sort(1, descending=True)

#         loss = self.rank_loss(scores, label_inds)

#         # Update metrics
#         self.metrics['loss'] += loss.item()
#         self.metrics['examples'] += batchsize
#         for b in range(batchsize):
#             rank = (ranks[b] == label_inds[b]).nonzero().item()
#             self.metrics['rank'] += 1 + rank

#         loss.backward()
#         self.update_params()

#         # Get predictions but not full rankings for the sake of speed
#         if cand_vecs.dim() == 2:
#             preds = [cands[ordering[0]] for ordering in ranks]
#         elif cand_vecs.dim() == 3:
#             preds = [cands[i][ordering[0]] for i, ordering in enumerate(ranks)]
#         return Output(preds)

#     def eval_step(self, batch): 
#         """Evaluate a single batch of examples."""
#         if batch.text_vec is None:
#             return
#         batchsize = batch.text_vec.size(0)
#         self.model.eval()

#         cands, cand_vecs, label_inds = self._build_candidates(
#             batch, source=self.opt['eval_candidates'], mode='eval')

#         scores = self.score_candidates(batch, cand_vecs)
#         _, ranks = scores.sort(1, descending=True)

#         # Update metrics
#         if label_inds is not None:
#             loss = self.rank_loss(scores, label_inds)
#             self.metrics['loss'] += loss.item()
#             self.metrics['examples'] += batchsize
#             for b in range(batchsize):
#                 rank = (ranks[b] == label_inds[b]).nonzero().item()
#                 self.metrics['rank'] += 1 + rank

#         cand_preds = []
#         for i, ordering in enumerate(ranks):
#             if cand_vecs.dim() == 2:
#                 cand_list = cands
#             elif cand_vecs.dim() == 3:
#                 cand_list = cands[i]
#             cand_preds.append([cand_list[rank] for rank in ordering])
#         preds = [cand_preds[i][0] for i in range(batchsize)]

#         if self.opt['interactive']:
#             return Output(preds)
#         else:
#             return Output(preds, cand_preds)
