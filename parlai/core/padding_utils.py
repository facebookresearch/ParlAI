import random
import torch

from collections import deque

class PaddingUtils(object):
    """
    Class that contains functions that help with padding input and target tensors.
    """
    @classmethod
    def pad_text(cls, observations, dictionary, end_idx, null_idx, dq=False, eval_labels=True, truncate=None):
        """We check that examples are valid, pad with zeros, and sort by length
           so that we can use the pack_padded function. The list valid_inds
           keeps track of which indices are valid and the order in which we sort
           the examples.
           dq -- whether we should use deque or list
           eval_labels -- whether or not we want to consider eval labels
           truncate -- truncate input and output lengths
        """
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs and len(obs['text']) > 0
        try:
            # valid examples and their indices
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None, None, None, None, None

        # set up the input tensors
        bsz = len(exs)

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        if any(['text2vec' in ex for ex in exs]):
            parsed_x = [ex['text2vec'] for ex in exs]
        else:
            parsed_x = [dictionary.txt2vec(ex['text']) for ex in exs]
        x_lens = [len(x) for x in parsed_x]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_x = [parsed_x[k] for k in ind_sorted]
        end_idxs = [x_lens[k] for k in ind_sorted]

        eval_labels_avail = any(['eval_labels' in ex for ex in exs])
        labels_avail = any(['labels' in ex for ex in exs])
        if eval_labels:
            some_labels_avail = eval_labels_avail or labels_avail
        else:
            some_labels_avail = labels_avail

        max_x_len = max(x_lens)

        # pad with zeros
        if dq:
            parsed_x = [x if len(x) == max_x_len else
                        x + deque((null_idx,)) * (max_x_len - len(x))
                        for x in parsed_x]
        else:
            parsed_x = [x if len(x) == max_x_len else
                        x + [null_idx] * (max_x_len - len(x))
                        for x in parsed_x]
        xs = torch.LongTensor(parsed_x)


        # set up the target tensors
        ys = None
        labels = None
        y_lens = None
        if some_labels_avail:
            # randomly select one of the labels to update on (if multiple)
            if labels_avail:
                labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            else:
                labels = [random.choice(ex.get('eval_labels', [''])) for ex in exs]
            # parse each label and append END
            if dq:
                parsed_y = [deque(maxlen=truncate) for _ in labels]
                for dq, y in zip(parsed_y, labels):
                    dq.extendleft(reversed(dictionary.txt2vec(y)))
            else:
                parsed_y = [dictionary.txt2vec(label) for label in labels]
            for y in parsed_y:
                y.append(end_idx)

            y_lens = [len(y) for y in parsed_y]
            max_y_len = max(y_lens)

            if dq:
                parsed_y = [y if len(y) == max_y_len else
                            y + deque((null_idx,)) * (max_y_len - len(y))
                            for y in parsed_y]
            else:
                parsed_y = [y if len(y) == max_y_len else
                            y + [null_idx] * (max_y_len - len(y))
                            for y in parsed_y]
            ys = torch.LongTensor(parsed_y)

        return xs, ys, labels, valid_inds, end_idxs, y_lens

    @classmethod
    def map_predictions(cls, predictions, valid_inds, batch_reply, observations, dictionary, end_idx, report_freq=0.1, labels=None, answers=None, ys=None):
        """Predictions are mapped back to appropriate indices in the batch_reply
           using valid_inds.
           report_freq -- how often we report predictions
        """
        if isinstance(predictions, torch.autograd.Variable):
            predictions = predictions.cpu().data
        else:
            predictions = predictions.cpu()
        for i in range(len(predictions)):
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a timelab
            curr = batch_reply[valid_inds[i]]
            output_tokens = []
            j = 0
            for c in predictions[i]:
                if c == end_idx and j!=0:
                    break
                else:
                    output_tokens.append(c)
                j+=1
            curr_pred = dictionary.vec2txt(output_tokens)
            curr['text'] = curr_pred

            if labels is not None and answers is not None:
                y = []
                for c in ys.data[i]:
                    if c == end_idx:
                        break
                    else:
                        y.append(c)
                answers[valid_inds[i]] = y
            elif answers is not None:
                answers[valid_inds[i]] = output_tokens


            if random.random() > (1 - report_freq):
                print('TEXT: ', observations[valid_inds[i]]['text'].replace('__END__', ''), '\n~')
                print('PREDICTION: ', curr_pred, '\n~')
        return
