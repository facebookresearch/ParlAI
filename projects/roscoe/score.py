#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import itertools
import math
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm import tqdm

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
except ImportError:
    raise ImportError('Please run `pip install transformers`.')

from projects.roscoe.utils import (
    cosine_similarity_scaled,
    embedding_alignment,
    al_mean,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Scorer needs sentence transformer model installed. \n "
        "pip install -U sentence-transformers"
    )
try:
    from simcse import SimCSE
except ImportError:
    raise ImportError(
        "Scorer needs SimCSE model installed. \n " "pip install -U simcse"
    )

SENT_TRANS = "sentence_transformer"
SIMSCE = "sim_sce"
SEQ_EMB_MODEL_TYPES = [SENT_TRANS, SIMSCE]

# sentence to word embedding models
TRANSFORMER_MODELS_DICT = {
    "all-mpnet-base-v2": "microsoft/mpnet-base",
    "all-MiniLM-L6-v2": "nreimers/MiniLM-L6-H384-uncased",
}
SIM_SCE_MODELS_DICT = {
    "princeton-nlp/sup-simcse-roberta-large": "roberta-large",
    "princeton-nlp/sup-simcse-roberta-base": "roberta-base",
    "princeton-nlp/sup-simcse-bert-base-uncased": "bert-base-uncased",
    "facebook/roscoe-512-roberta-base": "roberta-base",
}
EMBEDDING_MODELS = list(TRANSFORMER_MODELS_DICT.keys()) + list(
    SIM_SCE_MODELS_DICT.keys()
)
EMBEDDING_MODEL_NAMES = [m.split('/')[-1] for m in EMBEDDING_MODELS]

SCORE_GROUPS = ['semantic', 'inference', 'language']

# Cosine Similarity (CS): Cosine of angle between two vectors. Equivalently, dot product of two vectors divided by the
#   product of their magnitudes
# Alignment (A): Comparison of one text (ref_0..i) to another (hypo_0..j). Returns a vector (alignment_0..j), where k^th
#   component corresponds to a max cosine similarity score between hypo_k and (ref_0..i), i.e. confidence that the information
#   in j^th component in hypothesis is grounded by reference. alignment_j = [max_i(CS(ref_i, hypo_j))]
# Mean Alignment (MA): Given the alignment vecotor (alignment_0..j), return the average over all components. High scores
#   correspond to well-grounded hypothesis wrt reference. avg_j(max_i(CS(ref_i, hypo_j)))
# Inverse Alignment (IA): One minus the alignment scored (IA = 1 - A). High values in IA components correspond to the
#   components in hypothesis that are not grounded by the reference

# Mean Alignment from the hypothesis chain to to the context sentences. The higher the score, the better the hypothesis
# is grounded by the context.
FAITHFUL_SENT = "faithfulness"
# Mean Alignment of the sentence and token embeddings from the hypothesis chain to to the context sentences and tokens.
FAITHFUL_WORD = "faithfulness_ww"
# For each step in the chain, get the max MA of all the tokens from the current step's tokens to the tokens in previous
# steps. Returns the alignment score of the most aligned steps in the chain on the token level.
REPETITION_WORD = "repetition_word"
# Max of the cosine similarities of each step to all previous steps.
REPETITION_SENT = "repetition_step"
# Mean Alignment from the sentences in the context to all steps in the chain, and vice versa, averaged. (MA(x,y) + MA(y,x))/2
INFORM_STEP = "informativeness_step"
# Cosine similarity between the hypothesis embedding and context embedding as a whole
INFORM_CHAIN = "informativeness_chain"
# Uses NLI model to predict probability of contradiction of each step in hypothesis and each sentence in the context.
# Returns maximum of all these.
DISCOURSE_REPRESENTATION = "discourse_representation"
# Max probability of contradiction of each step in the chain to each of the previous steps. Predicted with an NLI model.
COHERENCE_STEP_VS_STEP = "coherence_step_vs_step"
# Perplexity of each step, averaged over the chain.
PPL_STEP = "perplexity_step"
# Maximum of the perplexities of each step, where each step is scored individually.
PPL_STEP_MAX = "perplexity_step_max"
# Perplexity of the whole chain taken as a continuous string
PPL_CHAIN = "perplexity_chain"
# Grammatical correctness of each step, as predicted by a grammaticality classifier, averaged over the chain.
GRAMMAR_STEP = "grammar_step"
# Grammatical correctness of each step, as predicted by a grammaticality classifier.
# Minimum score given to a step (most incorrect step's score is used).
GRAMMAR_STEP_MAX = "grammar_step_max"
# Mean Alignment from the hypothesis chain to the reference chain.
CHAIN_ALIGNMENT = "reasoning_alignment"
# IA from the hypothesis chain to the reference chain, multiplied element-wise by the IA from the hypothesis to the context.
# Returns max value that represents step in the hypothesis most irrelevant to the reference and context.
EXT_HALLUCINATION = "external_hallucination"
# Min Alignment from the hypothesis to the reference
REDUNDANCY = "redundancy"
# Like external_hallucination, but with IA from the reference to the hypothesis and to the context. Entry-wise product
# defines how much relevant information is missing in the hypothesis
COMMON_SENSE_ERROR = "common_sense_error"
# Minimum Alignment from the reference chain to the hypothesis chain.
MISSING_STEP = "missing_step"
# Absolute difference between MA from the reference chain to the context sentences and MA from the hypothesis chain
# to the context sentences.
SEMANTIC_COVERAGE_STEP = "semantic_coverage_step"
# Cosine similarity between hypothesis embedding and reference embedding as a whole.
SEMANTIC_COVERAGE_CHAIN = "semantic_coverage_chain"

UNSUPERVISED_SCORES = [
    FAITHFUL_SENT,
    FAITHFUL_WORD,
    REPETITION_WORD,
    REPETITION_SENT,
    INFORM_STEP,
    INFORM_CHAIN,
    DISCOURSE_REPRESENTATION,
    COHERENCE_STEP_VS_STEP,
    PPL_STEP,
    PPL_CHAIN,
    GRAMMAR_STEP,
    PPL_STEP_MAX,
    GRAMMAR_STEP_MAX,
]
SUPERVISED_SCORES = [
    CHAIN_ALIGNMENT,
    EXT_HALLUCINATION,
    REDUNDANCY,
    COMMON_SENSE_ERROR,
    MISSING_STEP,
    SEMANTIC_COVERAGE_STEP,
    SEMANTIC_COVERAGE_CHAIN,
]
REASONING_SCORES = UNSUPERVISED_SCORES + SUPERVISED_SCORES

ROSCOE_SA = [
    FAITHFUL_SENT,
    FAITHFUL_WORD,
    REPETITION_WORD,
    INFORM_STEP,
    CHAIN_ALIGNMENT,
    EXT_HALLUCINATION,
    REDUNDANCY,
    COMMON_SENSE_ERROR,
    MISSING_STEP,
    SEMANTIC_COVERAGE_STEP,
]

ROSCOE_SS = [
    INFORM_CHAIN,
    REPETITION_SENT,
    SEMANTIC_COVERAGE_CHAIN,
]

EMB_MODEL_SCORES = [
    FAITHFUL_SENT,
    FAITHFUL_WORD,
    REPETITION_WORD,
    REPETITION_SENT,
    INFORM_STEP,
    INFORM_CHAIN,
    CHAIN_ALIGNMENT,
    EXT_HALLUCINATION,
    REDUNDANCY,
    COMMON_SENSE_ERROR,
    MISSING_STEP,
    SEMANTIC_COVERAGE_STEP,
    SEMANTIC_COVERAGE_CHAIN,
]

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_MODEL_SCORES = [
    DISCOURSE_REPRESENTATION,
    COHERENCE_STEP_VS_STEP,
]
ROSCOE_LI = NLI_MODEL_SCORES

LANGUAGE_MODEL_SCORES = [
    PPL_CHAIN,
    PPL_STEP,
    PPL_STEP_MAX,
]

GRAMMAR_CHECK_MODEL_NAME = "cointegrated/roberta-large-cola-krishna2020"
GRAMMAR_MODEL_SCORES = [
    GRAMMAR_STEP,
    GRAMMAR_STEP_MAX,
]
ROSCOE_LC = LANGUAGE_MODEL_SCORES + GRAMMAR_MODEL_SCORES


class Chain:
    def __init__(self, line: List[str]) -> None:
        self.chain = self.parse_chain(line)
        self.sentence_embeddings = None
        self.whole_chain_embedding = None

    def parse_chain(self, chain: List[str]) -> List[str]:
        """
        Change formatting Returns list of steps in reasoning chain.
        """
        return chain

    def set_sentence_embeddings(self, embeddings: List[List[float]]):
        self.sentence_embeddings = embeddings

    def set_whole_chain_embedding(self, embeddings: List[float]):
        self.whole_chain_embedding = embeddings


def contains_embedding_scores(score_types: Iterable[str]) -> bool:
    return any(s in EMB_MODEL_SCORES for s in score_types)


def contains_nli_scores(score_types: Iterable[str]) -> bool:
    return any(s in NLI_MODEL_SCORES for s in score_types)


def contains_ppl_scores(score_types: Iterable[str]) -> bool:
    return any(s in LANGUAGE_MODEL_SCORES for s in score_types)


def contains_grammar_scores(score_types: Iterable[str]) -> bool:
    return any(s in GRAMMAR_MODEL_SCORES for s in score_types)


def select_semantic_scores(score_types: Iterable[str]) -> List[str]:
    return [s for s in score_types if s in EMB_MODEL_SCORES]


def select_language_scores(score_types: Iterable[str]) -> List[str]:
    return [s for s in score_types if s in GRAMMAR_MODEL_SCORES + LANGUAGE_MODEL_SCORES]


def select_inference_scores(score_types: Iterable[str]) -> List[str]:
    return [s for s in score_types if s in NLI_MODEL_SCORES]


class Evaluator:
    def __init__(
        self,
        hypos: List[Chain],
        context: List[Chain],
        references: Optional[List[Chain]] = None,
        score_types: List[str] = REASONING_SCORES,
        model_type: str = SENT_TRANS,
        transformer_model: str = "all-mpnet-base-v2",
        nli_model: str = NLI_MODEL_NAME,
        ppl_model: str = 'gpt2-large',
        grammar_model: str = GRAMMAR_CHECK_MODEL_NAME,
        discourse_batch: int = 64,
        coherence_batch: int = 8,
        ppl_batch: int = 16,
        grammar_batch: int = 8,
        random_seed: int = 42,
    ) -> None:
        # set seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.references = references
        self.hypos = hypos
        self.context = context

        self.score_types = score_types

        if contains_embedding_scores(score_types):
            self.sentence_model, self.word_model_name = self._create_model(
                model_type, transformer_model
            )
            # Load model from HuggingFace Hub
            if self.word_model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.word_model_name)
                self.model = AutoModel.from_pretrained(self.word_model_name)
                self.model.eval().to(self.device)
        if contains_nli_scores(score_types):
            # Load model for NLI-type predictions
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model
            )
            self.nli_model.eval().to(self.device)
            self.discourse_batch = discourse_batch
            self.coherence_batch = coherence_batch
        if contains_ppl_scores(score_types):
            # Load language model for PPL-based metrics
            self.ppl_model, self.ppl_tokenizer = self._build_ppl_model(
                model_id=ppl_model
            )
            self.cross_entropy_loss = nn.CrossEntropyLoss(
                reduction='none',
                ignore_index=self.ppl_tokenizer.pad_token_id,
            )
            self.ppl_batch = ppl_batch
        if contains_grammar_scores(score_types):
            # Load grammaticality model
            (
                self.grmr_model,
                self.grmr_tokenizer,
            ) = self._build_grammaticality_classifier(
                model_id=grammar_model,
            )
            self.grammar_batch = grammar_batch

        self._verify_input()

    def set_hypos(self, hypos: List[Chain]) -> None:
        self.hypos = hypos

    def set_references(self, references: List[Chain]) -> None:
        self.references = references

    def set_context(self, context: List[Chain]) -> None:
        self.context = context

    def _verify_input(self):
        if self.references and len(self.references) != len(self.hypos):
            raise ValueError(
                f"For each hypo there should be a reference. "
                + f"Found {len(self.references)} reference and {len(self.hypos)} hypothesis chains"
            )

    def _create_model(
        self, type: str, sentence_embedding_model: str
    ) -> Tuple[SentenceTransformer, str]:
        if type == SIMSCE and sentence_embedding_model in SIM_SCE_MODELS_DICT.keys():
            # should be auto-uploaded to CUDA if GPU is available
            return (
                SimCSE(sentence_embedding_model),
                SIM_SCE_MODELS_DICT[sentence_embedding_model],
            )
        elif (
            type == SENT_TRANS
            and sentence_embedding_model in TRANSFORMER_MODELS_DICT.keys()
        ):
            # Load model from HuggingFace Hub
            model = SentenceTransformer(sentence_embedding_model)
            model.eval().to(self.device)
            return (
                model,
                TRANSFORMER_MODELS_DICT[sentence_embedding_model],
            )
        else:
            raise ValueError(
                f"Only {list(TRANSFORMER_MODELS_DICT.keys())} and {list(SIM_SCE_MODELS_DICT.keys())} "
                + f"models are supported for {SEQ_EMB_MODEL_TYPES} sentence embeddings"
            )

    def _build_ppl_model(
        self, model_id: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
        if tokenizer.pad_token is None:
            # GPT2 doesn't have a pad token. This hack allows for padding a batch of tensors,
            # and the attention mask will exclude the pad tokens from the loss.
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(self.device)
        return model, tokenizer

    def _build_grammaticality_classifier(
        self,
        model_id: str,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = (
            AutoModelForSequenceClassification.from_pretrained(model_id)
            .eval()
            .to(self.device)
        )
        return model, tokenizer

    def embed_sentences(self, reasoning: List[str]) -> List[List[float]]:
        """Input:
            reasoning: list of strings (reasoning steps, for ex)
        Returns a list of embeddings, one per sentence
        """
        if len(reasoning) == 0:
            return [[]]
        embeddings = []
        with torch.no_grad():
            embeddings = self.sentence_model.encode(reasoning)
        # normalize
        if torch.is_tensor(embeddings[0]):
            embeddings = np.array([e.numpy() for e in embeddings])
        else:
            embeddings = np.array(embeddings)
        row_sums = np.sum(embeddings**2, axis=1)
        return embeddings / row_sums[:, np.newaxis]

    def embed_all_sentences(self, chains: List[Chain]):
        """
        Embed each sentence in each chain.
        """
        all_embeddings = self.embed_sentences(
            [sentence for chain in chains for sentence in chain.chain]
        )
        index = 0
        for chain in chains:
            if len(chain.chain) == 0:
                chain.set_sentence_embeddings([[]])
            else:
                embeddings = all_embeddings[index : index + len(chain.chain)]
                chain.set_sentence_embeddings(embeddings)
            index += len(chain.chain)

    def embed_all_chains(self, chains: List[Chain]):
        """
        Embed each chain as a whole.
        """
        all_embeddings = self.embed_sentences(
            [". ".join(chain.chain) for chain in chains]
        )
        for chain, embedding in zip(chains, all_embeddings):
            chain.set_whole_chain_embedding(embedding)

    def embed_words(self, reasoning: List[str]) -> List[List[float]]:
        """
        For each word in each step in the reasoning chain, return corresponding
        embedding.
        """
        if len(reasoning) == 0:
            return [[]]
        encoded_input = self.tokenizer(
            reasoning, truncation=True, padding=True, return_tensors="pt"
        )["input_ids"]
        with torch.no_grad():
            model_output = self.model(encoded_input.to(self.device))
        embeddings = []

        pads = (encoded_input == self.tokenizer.pad_token_id).nonzero()
        j = 0
        for i, emb in enumerate(model_output[0]):
            while j < len(pads) and pads[j][0] < i:
                j += 1
            if j < len(pads) and pads[j][0] == i:
                embeddings.append(emb[1 : pads[j][1] - 1].cpu().detach().numpy())
            else:
                embeddings.append(emb[1:-1].cpu().detach().numpy())

        return np.array(embeddings, dtype=object)

    def linearize_array(self, word_embeddings: np.array) -> np.array:
        """
        Input:
            [array([[-1.94962081e-02,  4.01699156e-01, -7.21625937e-03, ...,
                    3.15075251e-03, -1.78157743e-02, -4.47794721e-02],
                ...,
                [-1.24082603e-01,  6.57148898e-01, -6.30941940e-03, ...,
                    4.67915349e-02, -1.91840027e-02, -1.02941066e-01]], dtype=float32)
            array([[ 1.39849812e-01,  1.83899269e-01,  6.99776178e-03, ...,
                    7.79140592e-02, -1.22751370e-01,  4.21638079e-02],
                ...,
                [ 1.30717367e-01, -4.97479558e-01, -8.03395808e-02, ...,
                    -4.52864505e-02, -2.93156672e-02,  3.58024091e-02]], dtype=float32)]
        Output:
            [array([[-1.94962081e-02,  4.01699156e-01, -7.21625937e-03, ...,
                    3.15075251e-03, -1.78157743e-02, -4.47794721e-02],
                ...,
                ...,
                [ 1.30717367e-01, -4.97479558e-01, -8.03395808e-02, ...,
                    -4.52864505e-02, -2.93156672e-02,  3.58024091e-02]], dtype=float32)]
        """
        out = []
        for a in word_embeddings:
            out.extend(a)
        return out

    def repetitions(self, embeddings: np.array) -> float:
        """Input:
        3D-array: for each step, for each word contains an embedding vector
            [array([[-1.94962081e-02,  4.01699156e-01, -7.21625937e-03, ...,
                    3.15075251e-03, -1.78157743e-02, -4.47794721e-02],
                ...,
                [-1.24082603e-01,  6.57148898e-01, -6.30941940e-03, ...,
                    4.67915349e-02, -1.91840027e-02, -1.02941066e-01]], dtype=float32)
            array([[ 1.39849812e-01,  1.83899269e-01,  6.99776178e-03, ...,
                    7.79140592e-02, -1.22751370e-01,  4.21638079e-02],
                ...,
                [ 1.30717367e-01, -4.97479558e-01, -8.03395808e-02, ...,
                    -4.52864505e-02, -2.93156672e-02,  3.58024091e-02]], dtype=float32)]
            Returns:
                max_{t=1..n}(max_{k=0..t-1}[ RA(y_t -> y_k) ])

        OR
        2D-array: for each step, sentence embedding
        """
        assert len(embeddings) > 1
        words = False
        if len(embeddings[0].shape) == 2:
            words = True
        ra_most_similar_with_previous = []
        # compare steps' word embedding alignments
        for i in range(1, len(embeddings)):
            ra = 0
            # some embeddings can be empty. For example, for latex-style equations
            if len(embeddings[i]) > 0:
                # most similar sentence among previous sentences
                for j in range(i):
                    if len(embeddings[j]) > 0:
                        # word embeddings
                        if words:
                            ra = max(
                                ra,
                                al_mean(
                                    embedding_alignment(embeddings[j], embeddings[i])
                                ),
                            )
                        # sentence embeddings
                        else:
                            ra = max(
                                ra,
                                cosine_similarity_scaled(embeddings[j], embeddings[i]),
                            )

            ra_most_similar_with_previous.append(ra)

        return max(ra_most_similar_with_previous)

    def contradiction_probability(
        self, premise: Union[str, List[str]], hypothesis: Union[str, List[str]]
    ) -> List[float]:
        input = self.nli_tokenizer(
            premise, hypothesis, truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.nli_model(**input)
        prediction = torch.softmax(output["logits"], -1).tolist()

        return [p[2] for p in prediction]

    def max_contradiction(
        self, context: List[str], hypo_chain: List[str], batch_size: int
    ) -> float:
        # each hypo against each ref
        probs = []
        # split into batch size ~batch_size
        h_size = max(1, batch_size // len(context))
        all_pairs = list(itertools.product(context, hypo_chain))
        ind = 0
        while ind < len(all_pairs):
            cur_size = min(h_size, len(all_pairs) - ind)
            pr, hyp = zip(*all_pairs[ind : ind + cur_size])
            probs.extend(self.contradiction_probability(pr, hyp))
            ind += h_size
        return probs

    def discourse_represenation(
        self, context: List[str], hypo_chain: List[str], batch_size: int
    ) -> float:
        # each hypo against each ref
        probs = self.max_contradiction(context, hypo_chain, batch_size)
        return 1 - max(probs)

    def contradiction_step_vs_step(
        self, chain: List[str], batch_size: int
    ) -> List[float]:
        """
        Contradiction probability of each step.

        returns [ max_j { nli(y_k, y_j) } ]
        """
        return self.max_contradiction(chain, chain, batch_size)

    def cross_entropy(
        self, steps: List[str], batch_size: int = 16
    ) -> List[Tuple[float, int]]:
        """
        Cross-entropy and token count of each step.
        """
        results = []
        for i in range(0, len(steps), batch_size):
            losses_n_tokens = self.cross_entropy_batch(steps[i : (i + batch_size)])
            results.extend(losses_n_tokens)
        return results

    def cross_entropy_batch(self, batch: List[str]) -> List[Tuple[float, int]]:
        """
        Return cross-entropy across all tokens in batch, and the number of target tokens
        in this batch.
        """
        h_encodings = self.ppl_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
        )
        input_ids = h_encodings.input_ids
        attn_mask = h_encodings.attention_mask

        # find token count for each string, to be used for averaging later
        token_counts = attn_mask.sum(-1).tolist()

        # add start token
        bos_tokens_tensor = torch.tensor(
            [[self.ppl_tokenizer.bos_token_id]] * input_ids.size(dim=0)
        )
        input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)
        attn_mask = torch.cat(
            [
                torch.ones(bos_tokens_tensor.size(), dtype=torch.int64),
                attn_mask,
            ],
            dim=1,
        )

        losses = self.cross_entropy_strided(
            input_ids=input_ids, attention_mask=attn_mask
        )

        assert len(losses) == len(batch), "Losses should be separated by example."

        return list(zip(losses, token_counts))

    def cross_entropy_strided(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[float]:
        """
        Use for long strings of text that might not fit in the model's context.

        Technique described here: https://huggingface.co/docs/transformers/perplexity
        It scores "token_stride" tokens at a time using up to "max_length" tokens as context,
        straddling "i", and moving forward "token_stride" number of tokens at each step.
        Example with max_length = 4, on the second step, where tokens 2 through 4 are being scored:
        input_ids:             [1, 2, 3, 4, 5, 6]
        beg_loc, end_loc, i:    b     i     e
        """
        bsz = input_ids.size(0)
        # maximum context length the model is capable of processing
        max_length = self.ppl_model.config.n_positions
        assert max_length % 2 == 0
        token_stride = max_length // 2

        nlls = []
        end_loc = 0
        total_tokens_scored = 0
        # keep stepping forward until last token is scored
        for i in range(0, input_ids.size(1), token_stride):
            # score using max context possible
            beg_loc = max(i + token_stride - max_length, 0)
            end_loc = min(i + token_stride, input_ids.size(1))
            input = input_ids[:, beg_loc:end_loc].to(self.device)
            mask = attention_mask[:, beg_loc:end_loc].to(self.device)
            # for tokens we are only treating as context, set the target to ignore_index so they won't be scored
            target_len = end_loc - i
            label = input.clone()
            label[:, :-target_len] = self.cross_entropy_loss.ignore_index

            with torch.no_grad():
                outputs = self.ppl_model(input, attention_mask=mask, labels=label)

                # Copied from transformers GPT2LMHeadModel.forward implementation
                # so we can get unreduced losses:
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = label[..., 1:].contiguous()
                # Flatten the tokens
                loss = self.cross_entropy_loss(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                # unflatten
                loss = loss.view(bsz, -1)
                # first token is never scored, and masked tokens in the target sequence aren't scored
                new_tokens = mask[:, 1:][:, -target_len:].sum(-1)
                # accumulate loss, which will later be divided by the total token count
                neg_log_likelihood = loss.sum(-1)

            nlls.append(neg_log_likelihood)
            total_tokens_scored += new_tokens

        assert end_loc == input_ids.size(1), "we didn't get to the end of the sequence"
        assert (
            total_tokens_scored.sum() == attention_mask.sum() - bsz
        ), "our loss hasn't been weighted properly"

        return (torch.stack(nlls, dim=1).sum(-1) / total_tokens_scored).tolist()

    def perplexity(self, steps: List[str], batch_size: int) -> List[Tuple[float, int]]:
        """
        Perplexity and number of tokens of each step.

        Perplexity is in range [1, inf).
        """
        entropy_results = self.cross_entropy(steps, batch_size)
        return [(math.exp(e), n) for e, n in entropy_results]

    def check_grammar_batch(self, steps: List[str]) -> List[float]:
        """
        Uses a pre-trained grammar check model to score each step.

        Passes `steps` directly to the model, so should be called with a number of steps
        that will fit in memory.
        """
        encodings = self.grmr_tokenizer(
            steps,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.grmr_model(**encodings).logits
            probs = torch.softmax(logits, dim=-1)
        # 0=correct, 1=incorrect
        return probs[:, 0].tolist()

    def check_grammar(
        self, steps: List[str], batch_size: int = 8
    ) -> List[Tuple[str, float]]:
        """
        Uses a pre-trained grammar check model to score each step.
        """
        results = []
        for i in range(0, len(steps), batch_size):
            steps_batch = steps[i : (i + batch_size)]
            probs = self.check_grammar_batch(steps_batch)
            results.extend(zip(steps_batch, probs))
        return results

    def precompute_embeddings(
        self,
        hypos: List[Chain],
        contexts: List[Chain],
        references: Optional[List[Chain]],
    ) -> None:
        # embed all hypos using sentence embedder simultaneously
        self.embed_all_sentences(hypos)
        self.embed_all_sentences(contexts)
        self.embed_all_chains(hypos)
        self.embed_all_chains(contexts)
        if references:
            self.embed_all_sentences(references)
            self.embed_all_chains(references)

    def compute_embedding_scores(
        self,
        hypo: Chain,
        context: Chain,
        reference: Optional[Chain],
        score_types: List[str],
        scores: Dict[str, List],
    ) -> Dict[str, List]:
        h_steps_embeddings = hypo.sentence_embeddings
        c_steps_embeddings = context.sentence_embeddings
        h_whole_chain_embedding = hypo.whole_chain_embedding
        c_whole_chain_embedding = context.whole_chain_embedding
        # Sentence Embedding Matching between context and hypos
        y_x_sent_emb = embedding_alignment(
            ref_emb=c_steps_embeddings, hypo_emb=h_steps_embeddings
        )
        x_y_sent_emb = embedding_alignment(
            ref_emb=h_steps_embeddings, hypo_emb=c_steps_embeddings
        )

        if FAITHFUL_SENT in score_types:
            scores[FAITHFUL_SENT].append(al_mean(y_x_sent_emb))
        if INFORM_STEP in score_types:
            scores[INFORM_STEP].append(
                (al_mean(y_x_sent_emb) + al_mean(x_y_sent_emb)) / 2.0
            )
        if INFORM_CHAIN in score_types:
            scores[INFORM_CHAIN].append(
                (
                    1.0
                    + cosine_similarity_scaled(
                        h_whole_chain_embedding, c_whole_chain_embedding
                    )
                )
                / 2.0
            )
        if self.word_model_name:
            h_word_embeddings = self.embed_words(hypo.chain)
            c_word_embeddings = self.embed_words(context.chain)
            if FAITHFUL_WORD in score_types:
                scores[FAITHFUL_WORD].append(
                    al_mean(
                        np.append(
                            y_x_sent_emb,
                            embedding_alignment(
                                ref_emb=self.linearize_array(c_word_embeddings),
                                hypo_emb=self.linearize_array(h_word_embeddings),
                            ),
                        )
                    )
                )
            if REPETITION_WORD in score_types:
                # requires at least 2 sentences
                if len(h_word_embeddings) > 1:
                    scores[REPETITION_WORD].append(
                        1 - self.repetitions(h_word_embeddings)
                    )
                # otherwise give perfect score
                else:
                    scores[REPETITION_WORD].append(1.0)
        if REPETITION_SENT in score_types and self.word_model_name:
            # requires at least 2 sentences
            if len(h_steps_embeddings) > 1:
                scores[REPETITION_SENT].append(
                    (1.0 - self.repetitions(h_steps_embeddings)) / 2.0
                )
            # otherwise give perfect score
            else:
                scores[REPETITION_SENT].append(1.0)

        if reference and any(s in SUPERVISED_SCORES for s in score_types):
            r_steps_embeddings = reference.sentence_embeddings
            r_whole_chain_embedding = reference.whole_chain_embedding
            # Sentence Embedding Matching between references and hypos
            y_r_sent_emb = embedding_alignment(
                ref_emb=r_steps_embeddings, hypo_emb=h_steps_embeddings
            )
            r_y_sent_emb = embedding_alignment(
                ref_emb=h_steps_embeddings, hypo_emb=r_steps_embeddings
            )
            r_x_sent_emb = embedding_alignment(
                ref_emb=c_steps_embeddings, hypo_emb=r_steps_embeddings
            )
            if CHAIN_ALIGNMENT in score_types:
                scores[CHAIN_ALIGNMENT].append(al_mean(y_r_sent_emb))
            if EXT_HALLUCINATION in score_types:
                scores[EXT_HALLUCINATION].append(
                    1
                    - max(
                        np.multiply(
                            [1 - x for x in y_r_sent_emb],
                            [1 - x for x in y_x_sent_emb],
                        )
                    )
                )
            if REDUNDANCY in score_types:
                scores[REDUNDANCY].append(min(y_r_sent_emb))
            if COMMON_SENSE_ERROR in score_types:
                scores[COMMON_SENSE_ERROR].append(
                    1
                    - max(
                        np.multiply(
                            [1 - x for x in r_y_sent_emb],
                            [1 - x for x in r_x_sent_emb],
                        )
                    )
                )
            if MISSING_STEP in score_types:
                scores[MISSING_STEP].append(min(r_y_sent_emb))
            if SEMANTIC_COVERAGE_STEP in score_types:
                scores[SEMANTIC_COVERAGE_STEP].append(
                    1 - abs(al_mean(r_x_sent_emb) - al_mean(y_x_sent_emb))
                )
            if SEMANTIC_COVERAGE_CHAIN in score_types:
                scores[SEMANTIC_COVERAGE_CHAIN].append(
                    (
                        1.0
                        + cosine_similarity_scaled(
                            h_whole_chain_embedding, r_whole_chain_embedding
                        )
                    )
                    / 2.0
                )

        return scores

    def compute_nli_scores(
        self,
        context: Chain,
        hypo: Chain,
        score_types: List[str],
        scores: Dict[str, List],
    ):
        if DISCOURSE_REPRESENTATION in score_types:
            scores[DISCOURSE_REPRESENTATION].append(
                self.discourse_represenation(
                    context.chain, hypo.chain, self.discourse_batch
                )
            )

        if COHERENCE_STEP_VS_STEP in score_types:
            scores[COHERENCE_STEP_VS_STEP].append(
                # same batch as discourse
                (
                    1
                    - max(
                        self.contradiction_step_vs_step(
                            hypo.chain, self.discourse_batch
                        )
                    )
                )
            )
        return scores

    def compute_ppl_scores(
        self,
        hypo: Chain,
        score_types: List[str],
    ) -> Dict[str, float]:
        scores = {}

        if PPL_STEP in score_types or PPL_STEP_MAX in score_types:
            # Step-level Perplexities, for all scores that use them.
            step_perplexities = self.perplexity(hypo.chain, self.ppl_batch)

        def _token_weighted_mean(ppl_tkns_list):
            numer = sum(loss * n_tokens for loss, n_tokens in ppl_tkns_list)
            denom = sum(n_tokens for _, n_tokens in ppl_tkns_list)
            return numer / denom

        if PPL_STEP in score_types:
            scores[PPL_STEP] = 1 / _token_weighted_mean(step_perplexities)
        if PPL_CHAIN in score_types:
            chain_ppl_results = self.perplexity([' '.join(hypo.chain)], self.ppl_batch)
            chain_ppl = chain_ppl_results[0][0]
            scores[PPL_CHAIN] = 1 / chain_ppl
        if PPL_STEP_MAX in score_types:
            scores[PPL_STEP_MAX] = 1 / max(ppl for ppl, _ in step_perplexities)

        return scores

    def compute_grammar_scores(
        self,
        hypo: Chain,
        score_types: List[str],
    ) -> Dict[str, float]:
        scores = {}

        def _average_results(results):
            grammar_scores = [prob for _step, prob in results]
            return sum(grammar_scores) / len(grammar_scores)

        grammar_results = self.check_grammar(hypo.chain, self.grammar_batch)
        assert all(
            hypo_step == grmr_step
            for hypo_step, (grmr_step, _prob) in zip(hypo.chain, grammar_results)
        ), "steps were returned out of order"

        if GRAMMAR_STEP in score_types:
            scores[GRAMMAR_STEP] = _average_results(grammar_results)
        if GRAMMAR_STEP_MAX in score_types:
            # Minimum correctness score, i.e. maximum probability of error
            scores[GRAMMAR_STEP_MAX] = min(prob for _, prob in grammar_results)

        return scores

    def evaluate(self, score_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Score each reasoning chain.
        """
        score_types = score_types if score_types is not None else self.score_types
        # scores[score_type] = [score_chain1, score_chain2, ...]
        scores = defaultdict(list)

        if contains_embedding_scores(score_types):
            # Pre-compute chain, step, and word embeddings for embedding-based metrics
            self.precompute_embeddings(
                hypos=self.hypos,
                contexts=self.context,
                references=self.references,
            )

        for i in tqdm(range(len(self.hypos)), desc="Scoring chains ... "):
            if len(self.hypos[i].chain) == 0:
                for score in score_types:
                    scores[score].append("N/A")
                continue

            if contains_embedding_scores(score_types):
                # Calculate all embedding-based scores
                scores = self.compute_embedding_scores(
                    hypo=self.hypos[i],
                    context=self.context[i],
                    reference=self.references[i] if self.references else None,
                    score_types=score_types,
                    scores=scores,
                )

            if contains_nli_scores(score_types):
                scores = self.compute_nli_scores(
                    hypo=self.hypos[i],
                    context=self.context[i],
                    score_types=score_types,
                    scores=scores,
                )

            if contains_ppl_scores(score_types):
                ppl_scores = self.compute_ppl_scores(
                    hypo=self.hypos[i],
                    score_types=score_types,
                )
                for k, v in ppl_scores.items():
                    scores[k].append(v)

            # grammar acceptability classifier models
            if contains_grammar_scores(score_types):
                grammar_scores = self.compute_grammar_scores(
                    hypo=self.hypos[i],
                    score_types=score_types,
                )
                for k, v in grammar_scores.items():
                    scores[k].append(v)

        return scores
