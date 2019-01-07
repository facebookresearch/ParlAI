from parlai.core.torch_ranker_agent import TorchRankerAgent
from .bert_dictionary import BertDictionaryAgent
from .helpers import get_bert_optimizer, BertWrapper, BertModel, \
    add_common_args, surround
from parlai.core.utils import padded_3d
import torch
import json
import tqdm


class BiEncoderRankerAgent(TorchRankerAgent):
    """ TorchRankerAgent implementation of the biencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)

    def __init__(self, opt, shared=None):
        opt['rank_candidates'] = True
        opt["bert_id"] = 'bert-base-uncased'
        opt['candidates'] = "batch"
        if opt.get('eval_candidates', None) is None:
            opt['eval_candidates'] = "inline"
        self.clip = -1
        super().__init__(opt, shared)
        # NOTE: This is done AFTER init so that it's after load on purpose.
        # the state dict of MyModule and DataParallel(MyModule) is not the same
        if self.opt["multigpu"]:
            self.model = torch.nn.DataParallel(self.model)
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    def build_model(self):
        self.model = BiEncoderModule(self.opt)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        if all(f in self.opt for f in ["num_samples", "num_epochs", "batchsize"]):
            total_iterations = self.opt["num_samples"] * self.opt["num_epochs"] \
                / self.opt["batchsize"]
            self.optimizer = get_bert_optimizer([self.model],
                                                self.opt["type_optimization"],
                                                total_iterations,
                                                0.05,  # 5% scheduled warmup.
                                                self.opt["learningrate"])

    def receive_metrics(self, metrics_dict):
        """ Inibiting the scheduler.
        """
        pass

    def make_candidate_vecs(self, cands):
        cand_batches = [cands[i:i + 200] for i in range(0, len(cands), 200)]
        print("[ Vectorizing fixed candidates set from ({} batch(es) of up to 200) ]"
              "".format(len(cand_batches)))
        cand_vecs = []
        for batch in tqdm.tqdm(cand_batches):
            token_idx = [self._vectorize_text(cand, add_start=True, add_end=True,
                                              truncate=self.opt["token_cap"])
                         for cand in batch]
            padded_input = padded_3d([token_idx]).squeeze(0)
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                padded_input, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands)
            cand_vecs.append(embedding_cands.cpu().detach())
        return torch.cat(cand_vecs, 0)

    def vectorize(self, obs, add_start=True, add_end=True, truncate=None,
                  split_lines=False):
        return super().vectorize(
            obs,
            add_start=True,
            add_end=True,
            truncate=self.opt["token_cap"])

    def _set_text_vec(self, obs, truncate, split_lines):
        super()._set_text_vec(obs, truncate, split_lines)
        # concatenate the [CLS] and [SEP] tokens
        if obs is not None and "text_vec" in obs:
            obs["text_vec"] = surround(obs["text_vec"], self.START_IDX, self.END_IDX)
        return obs

    def score_candidates(self, batch, cand_vecs):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            batch.text_vec, self.NULL_IDX)
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None)

        if len(cand_vecs.size()) == 2 and cand_vecs.dtype == torch.long:
            # train time. We compare with all elements of the batch
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands)
            return embedding_ctxt.mm(embedding_cands.t())

        # predict time with multiple candidates
        if len(cand_vecs.size()) == 3:
            csize = cand_vecs.size()  # batchsize x ncands x sentlength
            cands_idx_reshaped = cand_vecs.view(csize[0] * csize[1], csize[2])
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cands_idx_reshaped, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands)
            embedding_cands = embedding_cands.view(
                csize[0], csize[1], -1)  # batchsize x ncands x embed_size
            embedding_cands = embedding_cands.transpose(
                1, 2)  # batchsize x embed_size x ncands
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            scores = torch.bmm(embedding_ctxt,
                               embedding_cands)  # batchsize x 1 x ncands
            scores = scores.squeeze(1)  # batchsize x ncands
            return scores

        # otherwise: cand_vecs should be 2D float vector ncands x embed_size
        return embedding_ctxt.mm(cand_vecs.t())


class BiEncoderModule(torch.nn.Module):
    """ Groups context_encoder and cand_encoder together.
    """

    def __init__(self, opt):
        super(BiEncoderModule, self).__init__()
        self.context_encoder = BertWrapper(
            BertModel.from_pretrained(
                opt["bert_id"]),
            opt["out_dim"],
            add_transformer_layer=opt["add_transformer_layer"],
            layer_pulled=opt["pull_from_layer"])
        self.cand_encoder = BertWrapper(
            BertModel.from_pretrained(
                opt["bert_id"]),
            opt["out_dim"],
            add_transformer_layer=opt["add_transformer_layer"],
            layer_pulled=opt["pull_from_layer"])

    def forward(self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
                token_idx_cands, segment_idx_cands, mask_cands):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands)
        return embedding_ctxt, embedding_cands

    def save(self, path=None):
        """ The state dict of DataParallel(MyModule) is not the same as
            MyModule. In the case where we use multigpu, save it differently.
        """
        if not self.multigpu:
            return super().save(self, path)
        path = self.opt.get('model_file', None) if path is None else path
        if path:
            states = {}
            states['model'] = self.model.module.state_dict()
            states['optimizer'] = self.optimizer.state_dict()
            with open(path, 'wb') as write:
                torch.save(states, write)
            # save opt file
            with open(path + '.opt', 'w') as handle:
                if hasattr(self, 'model_version'):
                    self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = (token_idx != null_idx).long()
    token_idx = token_idx * mask  # nullify elements in case self.NULL_IDX was not 0
    return token_idx, segment_idx, mask
