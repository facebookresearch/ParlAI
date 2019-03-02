#!/usr/bin/env python

import timeit
import torch
import apex.fp16_utils.fp16_optimizer as fp16_optimizer
import apex.optimizers.fused_adam as fused_adam
import argparse
import torch.nn.functional as F
import torch.utils.data
from parlai.core.utils import AttrDict
from parlai.agents.transformer.modules import TransformerEncoder as ParlaiTransformerEncoder
from fairseq.models.transformer import TransformerEncoder as FSTransformerEncoder

class SimpleModel(torch.nn.Module):
    def __init__(self, embedding, esz, ffn):
        super().__init__()
        self.embedding = embedding
        self.lin = torch.nn.Linear(esz, ffn)
        self.lin2 = torch.nn.Linear(ffn, esz)

    def forward(self, x):
        return self.lin2(F.relu(self.lin(self.embedding(x))))

NUM_BATCHES = 300
BS = 32
SEQLEN = 512
VOCAB = 16384
FFN = 1024
ESZ = 256
N_LAYERS = 4
N_HEADS = 2

ap = argparse.ArgumentParser()
ap.add_argument('--unsafe', action='store_true')
ap.add_argument('--bs', default=BS, type=int, help='batch size')
ap.add_argument('--type', default='parlai', choices={'parlai', 'fairseq', 'linear'})
ap.add_argument('--seqlen', default=SEQLEN, type=int, help='Sequence length')
ap.add_argument('--esz', default=ESZ, type=int, help='embedding size')
ap.add_argument('--ffn', default=FFN, type=int, help='ffn size')
ap.add_argument('--layers', type=int, default=N_LAYERS, help='mha layers')
ap.add_argument('--heads', type=int, default=N_HEADS, help='mha heads')
ap.add_argument('--vocab', type=int, default=VOCAB, help='vocab size')
ap.add_argument('--num', type=int, default=NUM_BATCHES, help='number of iterations')

args = ap.parse_args()


def build_model(args):
    embeddings = torch.nn.Embedding(args.vocab, args.esz, 0)
    if args.type == 'parlai':
        model = ParlaiTransformerEncoder(
            args.heads,  # heads
            args.layers,  # layers
            args.esz,  # embedding size
            args.ffn,  # ffn size
            args.vocab,  # vocab size
            embedding=embeddings,
            learn_positional_embeddings=False,
            n_positions=args.seqlen,
            reduction=False,
        )
    elif args.type == 'fairseq':
        model = FSTransformerEncoder(AttrDict({
            'dropout': 0.0,
            'max_source_positions': args.seqlen - 1,  # fairseq adds 1
            'encoder_learned_pos': False,
            'encoder_normalize_before': False,
            'encoder_layers': args.layers,
            'encoder_attention_heads': args.heads,
            'encoder_relu_dropout': 0.0,
            'encoder_ffn_embed_dim': args.ffn,
            'no_token_positional_embeddings': False,
            'encoder_embed_dim': args.esz,
            'attention_dropout': 0.0,
            'relu_dropout': 0.0,
        }), None, embeddings)
    elif args.type == 'linear':
        model = SimpleModel(embeddings, args.esz, args.ffn)
    return model, embeddings



class RandomSentences(torch.utils.data.Dataset):
    def __init__(self, args):
        self.seqlen = args.seqlen
        self.vocab = args.vocab

    def __len__(self):
        return BS * NUM_BATCHES

    def __getitem__(self, i):
        return torch.randint(low=1, high=self.vocab - 3, size=(self.seqlen, ), dtype=torch.int64)

    @classmethod
    def collate_fn(cls, batch):
        return torch.stack(batch).cuda()



dl = torch.utils.data.DataLoader(
    RandomSentences(args),
    args.bs,
    collate_fn=RandomSentences.collate_fn,
    num_workers=0
)

def run_loop(dl, args, use_fp16, use_apex):
    model, embeddings = build_model(args)
    lens = torch.LongTensor([args.seqlen] * args.bs).cuda()
    model = model.cuda()
    if use_fp16:
        model = model.half()
    else:
        model = model.float()

    for p in model.parameters():
        for d in p.shape:
            if d % 8 != 0:
                print("Warning: parameter isn't mod 8: ", p.shape)

    if use_apex:
        opt = fused_adam.FusedAdam(model.parameters(), lr=1e-5)
        opt = fp16_optimizer.FP16_Optimizer(opt) #, dynamic_loss_scale=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    start = timeit.default_timer()
    for x in dl:
        opt.zero_grad()

        if args.type == 'parlai':
            enc, _ = model(x)
        elif args.type == 'fairseq':
            enc = model(x, lens)['encoder_out']
        elif args.type == 'linear':
            enc = model(x)

        output = F.linear(enc.view(-1, enc.size(2)), embeddings.weight)
        output = output.view(-1, output.size(-1))
        if args.unsafe:
            logits = F.log_softmax(output, dim=-1)
        else:
            logits = F.log_softmax(output.float(), dim=-1).type_as(output)
        loss = F.nll_loss(logits, x.view(-1) + 1)

        if use_apex:
            opt.backward(loss)
            opt.clip_master_grads(0.1)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()

    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(loss.item())

    del model
    del embeddings
    return end - start

fp16_apex = run_loop(dl, args, True, True)
print("FP16, APEX: %.2f" % fp16_apex)
fp16_raw = run_loop(dl, args, True, False)
print("FP16, RAW: %.2f" % fp16_raw)
fp32 = run_loop(dl, args, False, False)
print("FP32:", fp32)
print("Speedup: %.1f" % (fp32 / fp16_apex))
