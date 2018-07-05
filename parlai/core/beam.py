import torch
import math

class Beam(object):
    def __init__(self, beam_size, min_length=3, padding_token=0, bos_token=1, eos_token=2, min_n_best=3, cuda=False):
        """
        Generic beam class. It keeps information about beam_size hypothesis.
        :param beam_size: number of hypothesis in the beam
        :param min_length: minimum length of the predicted sequence
        :param padding_token: Set to 0 as usual in ParlAI
        :param bos_token: Set to 1 as usual in ParlAI
        :param eos_token: Set to 2 as usual in ParlAI
        :param min_n_best: Beam will not be done unless this amount of finished hypothesis (with EOS) is done
        :param cuda: What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        if cuda is True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(self.device)  # recent score for each hypo in the beam
        self.all_scores = [torch.Tensor([0.0]*beam_size).to(self.device)]  # self.scores values per each time step
        self.bookkeep = []  # backtracking id to hypothesis at previous time step
        self.outputs = [torch.Tensor(self.beam_size).long().fill_(padding_token).to(self.device)]  # output tokens at each time step
        self.finished = []  # keeps tuples (score, time_step, hyp_id)
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        voc_size = softmax_probs.size(-1)

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
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = best_scores
        self.all_scores.append(self.scores)
        hyp_id = best_idxs / voc_size
        tok_id = best_idxs % voc_size

        self.outputs.append(tok_id)
        self.bookkeep.append(hyp_id)

        #  check new hypos for eos label, if we have some, add to finished
        for i in range(self.beam_size):
            if self.outputs[-1][i] == self.eos:
                #  this is finished hypo, adding to finished
                self.finished.append((self.scores[i], len(self.outputs)-1, i))
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Helper function to get single best hypothesis
        :return: hypothesis sequence represented as t{}-w{}-sc{:.{prec}f}
        """
        top_endtok = self.get_rescored_finished(n_best=1)[0]
        return self.get_hyp_from_finished(top_endtok[1], top_endtok[2]), top_endtok[0]

    def get_hyp_from_finished(self, timestep, hyp_id):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id
        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence represented as t{}-w{}-sc{:.{prec}f}
        """
        hypo_exist = False
        for finished_item in self.finished:
            if timestep == finished_item[1] and hyp_id == finished_item[2]:
                hypo_exist = True
        if hypo_exist == False:
            raise RuntimeError("There is no requested hypo in self.finished: tstep:{}, hypid:{}".format(timestep, hyp_id
                                                                                                        ))
        hyp_idx = []
        endback = hyp_id
        for i in range(timestep, -1, -1):
            hyp_idx.append('t{}-w{}-sc{:.{prec}f}'.format(i, self.outputs[i][endback], self.all_scores[i][endback], prec=3))
            endback = self.bookkeep[i-1][endback]

        return hyp_idx

    def get_rescored_finished(self, n_best=None):
        """

        :param n_best: how many n best hypothesis to return
        :return: list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item[1] + 1  # timestep + 1
            length_penalty = math.pow((1 + current_length)/6, 0.65)
            rescored_finished.append((finished_item[0]/length_penalty, finished_item[1], finished_item[2]))

        srted = sorted(rescored_finished, reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def get_beam_dot(self, dictionary=None, n_best=4):
        """
        Creates pydot graph representation of the beam
        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :return: pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for endtok in sorted_finished:
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(endtok[1], endtok[2]))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                idx = 't{}-w{}-sc{:.{prec}f}'.format(tstep, token, all_scores[tstep][hypid], prec=3)
                color = 'white'
                rank = None
                for i,hypseq in enumerate(top_hyp_idx_n_best):
                    if idx in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = "<{}".format(dictionary.vec2txt([token]) if dictionary is not None else token) + " : " + "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3)
                graph.add_node(pydot.Node(idx, label=label, fillcolor=color, style='filled', xlabel='{}'.format(rank) if rank is not None else '' ))

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i,prev_id in enumerate(lis):
                to_idx = '"t{}-w{}-sc{:.{prec}f}"'.format(revtstep+1, outputs[revtstep+1][i], all_scores[revtstep+1][i], prec=3)
                from_idx = '"t{}-w{}-sc{:.{prec}f}"'.format(revtstep, outputs[revtstep][prev_id], all_scores[revtstep][prev_id], prec=3)
                from_node = graph.get_node(from_idx)[0]
                to_node = graph.get_node(to_idx)[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph


