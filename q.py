#!/usr/bin/env python3

import sys
import pdb
import queue
import torch
import time
import random
import torch.multiprocessing as multiprocessing
import tokenizers
from transformers import GPT2Tokenizer, GPT2TokenizerFast

NUM_EXAMPLES = 200
NUM_WORKERS = 10
BS = 32
SEQ = 128


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class End:
    pass


def tok(tokenizer, text):
    r = {
        'field1': tokenizer.encode(text, return_tensors='pt')[0, :128],
        'field2': tokenizer.encode(text, return_tensors='pt')[0, :128],
        # 'field3': tokenizer.encode(text, return_tensors='pt')[0, :128],
        # 'field4': tokenizer.encode(text, return_tensors='pt')[0, :128],
    }

    r['len'] = len(r['field1'])
    return r


def generate_tensors(worker_id, queue, evt):
    print(f"Starting generate")
    tokenizer = get_tokenizer()
    sent_buffer = False
    buff = None
    ongoing = []
    with open("bible.txt") as f:
        for i, line in enumerate(f.readlines()):
            ongoing.append(line)
            if len(ongoing) > 10:
                ongoing.pop(0)
            if i % NUM_WORKERS == worker_id:
                text = "\n".join(ongoing)
                tokenized = tok(tokenizer, text)
                if buff is None:
                    buff = tokenized
                    for k in buff.keys():
                        if k == 'len':
                            continue
                        tokenized[k].share_memory_()
                else:
                    for k in buff.keys():
                        if k == 'len':
                            continue
                        if len(buff[k]) >= len(tokenized[k]):
                            buff[k].zero_()
                            buff[k][: len(tokenized[k])] = tokenized[k]
                            sent_buffer = False
                        else:
                            buff[k] = tokenized[k].share_memory_()
                if sent_buffer:
                    queue.put((worker_id, None))
                else:
                    sent_buffer = True
                    queue.put((worker_id, buff))
    queue.put((worker_id, End()))
    evt.wait()


def get_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")


def generate_tensors_slow():
    print(f"Starting slow generate")
    tokenizer = get_tokenizer()
    with open("bible.txt") as f:
        ongoing = []
        for i, line in enumerate(f.readlines(), 1):
            ongoing.append(line)
            if len(ongoing) > 10:
                ongoing.pop(0)
            text = "\n".join(ongoing)
            if i % 10000 == 0:
                print(i)
            item = tok(tokenizer, text)
            item = {k: v.cuda() if k != 'len' else v for k, v in item.items()}
            yield item


class Reader(object):
    def __init__(self):
        mp = multiprocessing.get_context("spawn")
        self.queue = mp.Manager().Queue()
        self.events = [mp.Event() for _ in range(NUM_WORKERS)]
        self.buffers = [None] * NUM_WORKERS
        self.processes = [
            mp.Process(target=generate_tensors, args=(i, self.queue, self.events[i]))
            for i in range(NUM_WORKERS)
        ]
        self.i = 0
        self.running_workers = 0
        for p in self.processes:
            p.start()
            self.running_workers += 1
        print("Done initializing")

    def _finish(self, worker_id):
        self.running_workers -= 1

    def __del__(self):
        for e in self.events:
            e.set()
        for p in self.processes:
            p.terminate()

    def parley(self):
        example = None
        while example is None:
            worker_id, example = self.queue.get()
            if isinstance(example, End):
                self._finish(worker_id)
                if self.running_workers == 0:
                    raise StopIteration
                example = None
            elif example is None:
                example = self.buffers[worker_id]
            else:
                self.buffers[worker_id] = example
        cuda = {k: v.cuda() if k != 'len' else v for k, v in example.items()}

        self.i += 1
        if self.i % 10000 == 0:
            print(self.i)


def main():
    import timeit

    print("slow")
    dt = timeit.default_timer()
    for item in generate_tensors_slow():
        pass
    print(timeit.default_timer() - dt)

    print('"fast"')
    r = Reader()

    dt = timeit.default_timer()
    while True:
        try:
            r.parley()
        except StopIteration:
            break
    print(timeit.default_timer() - dt)


if __name__ == '__main__':
    main()
