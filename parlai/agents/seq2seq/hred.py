#!/usr/bin/env python3
import torch
import torch.nn as nn

from .modules import (
    AttentionLayer,
    OutputLayer,
    RNNEncoder,
    Seq2seq,
    _transpose_hidden_state,
    opt_to_kwargs,
)
from .seq2seq import Seq2seqAgent


class HredModel(Seq2seq):
    """
    HRED model that has a context LSTM layer.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        device,
        numlayers=2,
        dropout=0,
        bidirectional=False,
        rnn_class="lstm",
        lookuptable="unique",
        decoder="same",
        numsoftmax=1,
        attention="none",
        attention_length=48,
        attention_time="post",
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):

        super(HredModel.__bases__[0], self).__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )

        rnn_class = nn.LSTM
        embedding_size = embeddingsize
        hidden_size = hiddensize
        # has to be set for super().reorder_encoder_states
        self.attn_type = "none"

        self.encoder = HredEncoder(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
            rnn_class=rnn_class,
            numlayers=numlayers,
            device=device,
        )

        self.decoder = HredDecoder(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
            rnn_class=rnn_class,
            numlayers=numlayers,
        )

        self.output = OutputLayer(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, hidden, attn_mask, context_vec = encoder_states
        enc_out, hidden, attn_mask = super().reorder_encoder_states(
            (enc_out, hidden, attn_mask), indices
        )
        context_vec = context_vec.index_select(0, indices)
        return enc_out, hidden, attn_mask, context_vec


class HredAgent(Seq2seqAgent):
    def __init__(self, opt, shared=None):
        """
        Set up model.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        super().__init__(opt, shared)
        self.id = "Hred"

    def build_model(self, states=None):
        opt = self.opt
        if not states:
            states = {}
        kwargs = opt_to_kwargs(opt)

        model = HredModel(
            len(self.dict),
            opt["embeddingsize"],
            opt["hiddensize"],
            device=self.device,
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get("longest_label", 1),
            **kwargs,
        )

        if opt.get("dict_tokenizer") == "bpe" and opt["embedding_type"] != "random":
            print("skipping preinitialization of embeddings for bpe")
        elif not states and opt["embedding_type"] != "random":
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt["embedding_type"])
            if opt["lookuptable"] in ["unique", "dec_out"]:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.encoder.lt.weight, opt["embedding_type"], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states["model"])

        if opt["embedding_type"].endswith("fixed"):
            print("Seq2seq: fixing embedding weights.")
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt["lookuptable"] in ["dec_out", "all"]:
                model.output.weight.requires_grad = False

        return model

    def batchify(self, obs_batch, sort=True):
        """
        Add action and attribute supervision for batches.
        Store history vec as context_vec.
        """
        batch = super().batchify(obs_batch)
        batch["context_vec"], batch["hist_lens"] = self.parse_context_vec(batch)
        return batch

    def parse_context_vec(self, batch):
        batch_context_vec = []
        hist_lens = []
        for i in range(len(batch["observations"])):
            hist_len = len(batch["observations"][i]["context_vec"])
            hist_lens.append(hist_len)
            for j in range(hist_len):
                context_vec = batch["observations"][i]["context_vec"][j]
                batch_context_vec.append(torch.tensor(context_vec, device=self.device))

        padded_context_vec = torch.nn.utils.rnn.pad_sequence(
            batch_context_vec, batch_first=True
        ).squeeze(1)
        return (
            padded_context_vec,
            torch.tensor(hist_lens, dtype=torch.long, device=self.device),
        )

    def _model_input(self, batch):
        return (batch.text_vec, batch.context_vec, batch.hist_lens)

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.
        Overridden to include both local utterance (text_vec) and full history (context_vec)
        """
        if "text" not in obs:
            return obs

        if "text_vec" not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs["full_text"] = history_string
            if history_string:
                history_vec = history.get_history_vec_list()
                obs["text_vec"] = history_vec[-1]
                obs["context_vec"] = history_vec

        # check truncation
        if obs.get("text_vec") is not None:
            truncated_vec = self._check_truncate(obs["text_vec"], truncate, True)
            obs.force_set("text_vec", torch.LongTensor(truncated_vec))
        return obs

    def _dummy_batch(self, batchsize, maxlen):
        """
        Overridden to add dummy context vec and hist lens.
        """
        batch = super()._dummy_batch(batchsize, maxlen)
        batch["context_vec"] = batch["text_vec"]
        batch["hist_lens"] = torch.ones(batchsize, dtype=torch.long)
        return batch


class HredEncoder(RNNEncoder):
    """
    RNN Encoder. Modified to encode history vector in context lstm.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        device,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidirectional=False,
        shared_lt=None,
        shared_rnn=None,
        input_dropout=0,
        unknown_idx=None,
        sparse=False,
    ):
        """
        Initialize recurrent encoder and context lstm.
        """
        super().__init__(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx,
            rnn_class,
            numlayers,
            dropout,
            bidirectional,
            shared_lt,
            shared_rnn,
            input_dropout,
            unknown_idx,
            sparse,
        )
        self.padding_idx = padding_idx
        self.context_lstm = nn.LSTM(
            hiddensize, hiddensize, numlayers, batch_first=True
        ).to(device)

    def forward(self, xs, context_vec, hist_lens):
        # encode current utterrance
        (enc_state, (hidden_state, cell_state), attn_mask) = super().forward(xs)

        # get utt lengths of each utt in context vector
        utt_lens = torch.sum(context_vec.ne(0).int(), dim=1)
        # sort by lengths descending for utterance encoder
        sorted_lens, sorted_idx = utt_lens.sort(descending=True)
        (_, (sorted_hidden_state, _), _) = super().forward(context_vec[sorted_idx])
        sorted_final_hidden_states = sorted_hidden_state[:, -1, :]

        # return encoded context_vec utterances to original order
        original_order_final_hidden = torch.zeros_like(
            sorted_final_hidden_states
        ).scatter_(
            0,
            sorted_idx.unsqueeze(1).expand(-1, sorted_final_hidden_states.shape[1]),
            sorted_final_hidden_states,
        )
        # reshape and pad encoded utterrances to bsz x max_seq_len x hidden_size using hist_lens
        original_size_final_hidden = self.sequence_to_padding(
            original_order_final_hidden, hist_lens
        )
        # pack padded sequence
        original_size_final_hidden_packed = nn.utils.rnn.pack_padded_sequence(
            original_size_final_hidden,
            hist_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # pass through context_lstm
        _, (context_h_n, _) = self.context_lstm(original_size_final_hidden_packed)
        return (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            _transpose_hidden_state(context_h_n),
        )

    def sequence_to_padding(self, x, lengths):
        """
        Return padded and reshaped sequence (x) according to tensor lengths
        Example:
            x = tensor([[1, 2], [2, 3], [4, 0], [5, 6], [7, 8], [9, 10]])
            lengths = tensor([1, 2, 2, 1])
        Would output:
            tensor([[[1, 2], [0, 0]],
                    [[2, 3], [4, 0]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [0, 0]]])
        """
        ret_tensor = torch.zeros(
            (lengths.shape[0], torch.max(lengths).int()) + tuple(x.shape[1:])
        ).to(x.device)
        cum_len = 0
        for i, l in enumerate(lengths):
            ret_tensor[i, :l] = x[cum_len : cum_len + l]
            cum_len += l
        return ret_tensor


class HredDecoder(nn.Module):
    """
    Recurrent decoder module that uses dialog history encoded by context lstm.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidir_input=False,
        attn_length=-1,
        sparse=False,
    ):
        """
        Initialize recurrent decoder.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize

        self.lt = nn.Embedding(
            num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse
        )
        self.rnn = rnn_class(
            embeddingsize + hiddensize,
            hiddensize,
            numlayers,
            dropout=dropout if numlayers > 1 else 0,
            batch_first=True,
        )

    def forward(self, xs, encoder_output, incremental_state=None):
        """
        Decode from input tokens.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from HredEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask, context_hidden) tuple.
        :param incremental_state: most recent hidden state to the decoder.

        :returns: (output, hidden_state) pair from the RNN.

            - output is a bsz x time x latentdim matrix. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            context_hidden,
        ) = encoder_output

        # sequence indices => sequence embeddings
        seqlen = xs.size(1)
        xes = self.dropout(self.lt(xs))

        # concatentate context lstm hidden state
        context_hidden_final_layer = context_hidden[:, -1, :].unsqueeze(1)
        resized_context_h = context_hidden_final_layer.expand(-1, seqlen, -1)
        xes = torch.cat((xes, resized_context_h), dim=-1).to(xes.device)

        # run through rnn with None as initial decoder state
        # source for zeroes hidden state: http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf
        output, new_hidden = self.rnn(xes, None)

        return output, _transpose_hidden_state(new_hidden)
