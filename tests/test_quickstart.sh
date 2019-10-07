# view a task & train a model
python examples/display_data.py -t babi:task10k:1
python examples/train_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -bs 1 -nt 4 -eps 5 -m memnn --no-cuda
python examples/display_model.py -t babi:task10k:1 -mf /tmp/babi_memnn -ecands vocab

# train a transformer on twitter
pip3 install emoji unidecode
python examples/display_data.py -t twitter
python examples/train_model.py -t twitter -mf /tmp/tr_twitter -m transformer/ranker -bs 10 -vtim 3600 -cands batch -ecands batch --data-parallel True --num-epochs 0.01
python examples/eval_model.py -t twitter -m transformer/ranker -mf models:convai2/seq2seq/convai2_self_seq2seq_model
python examples/display_model.py -t twitter -mf /tmp/tr_twitter -ecands batch

# add a simple model
mkdir parlai/agents/parrot
touch parlai/agents/parrot/parrot.py
echo "from parlai.core.torch_agent import TorchAgent, Output" >> parlai/agents/parrot/parrot.py
echo "class ParrotAgent(TorchAgent):" >> parlai/agents/parrot/parrot.py
echo "    def train_step(self, batch):" >> parlai/agents/parrot/parrot.py
echo "        pass" >> parlai/agents/parrot/parrot.py
echo "    def eval_step(self, batch):" >> parlai/agents/parrot/parrot.py
echo "        return Output([self.dict.vec2txt(row) for row in batch.text_vec])" >> parlai/agents/parrot/parrot.py
python examples/display_model.py -t babi:task10k:1 -m parrot
python examples/build_dict.py -t babi:task10k:1 -df /tmp/parrot.dict
python examples/display_model.py -t babi:task10k:1 -m parrot -df /tmp/parrot.dict
