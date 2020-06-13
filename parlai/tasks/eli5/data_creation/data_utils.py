#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Taken from
https://github.com/facebookresearch/ELI5/blob/master/data_creation/data_utils.py.
"""

import json
import math
import re
import signal

from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin

try:
    from spacy.lang.en import English
except ImportError:
    raise ImportError(
        "Please install spaCy: \n pip install -U spacy \n"
        "or follow installation instructions found at spacy.io/usage"
    )


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("timed_out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# URL match regex
URL_REGEX = (
    r"?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|"
    r"redu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|"
    r"post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|"
    r"aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|"
    r"cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|"
    r"do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|"
    r"gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|"
    r"io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|"
    r"li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|"
    r"mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|"
    r"pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|"
    r"sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|"
    r"tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|"
    r"vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]"
    r"+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|"
    r"""\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+"""
    r"(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|"
    r"coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|"
    r"af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|"
    r"bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|"
    r"cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|"
    r"fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|"
    r"gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|"
    r"kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|"
    r"me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|"
    r"ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|"
    r"qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|"
    r"st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|"
    r"ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
)

html_pairs = [
    ("&amp;", " & "),
    ("&quot", ' " '),
    ("&apos", " ' "),
    ("&gt;", " > "),
    ("&lt;", " < "),
]

tokenizer = English().Defaults.create_tokenizer()


# tokenizes and removes URLs (kept in separate list)
def pre_word_url_tokenize(stp):
    url_list = list(set(re.findall(URL_REGEX, stp)))
    # stp = st.strip()
    for i, url in enumerate(url_list):
        stp = stp.replace(url, " URL_%d " % (i,))
    for a, b in html_pairs:
        stp = stp.replace(a, b)
    pre_txt = ' '.join([str(x) for x in tokenizer(stp)])
    return (' '.join(pre_txt.split()), url_list)


# wrap inside a timer to catch cases where SpaCy tokenizer hangs on too many dots
def word_url_tokenize(st, max_len=20480, max_cont_len=512):
    stp = ' '.join(
        [
            w[:max_cont_len] if w[:max_cont_len].count('.') <= 12 else '.'
            for w in st.split()[:max_len]
        ]
    )
    try:
        with time_limit(2):
            return pre_word_url_tokenize(stp)
    except TimeoutException:
        print('timeout', len(st), ' --n-- '.join(st[:128].split('\n')))
        print(' --n-- '.join(stp.split('\n')))
        res = ('missed page', [])
        return res


# split tokenized text into sentences, split sentences that are too long
# ignores sentences with fewer than 2 words
def sentence_split(st, max_len=120, max_sen=-1):
    pre_sentences = st.split('\n')
    res = []
    for pre_s in pre_sentences:
        add_sent = ' '.join(
            [
                x + '</s>'
                if (
                    x in ['.', '!', '?'] or '.[' in x
                )  # .[ for tokenizer failure on wiki
                else x
                for x in pre_s.split()
            ]
        )
        pre_res = add_sent.split('</s>')
        pre_lst = []
        # aggressively cut down long sentences
        for sen in pre_res:
            if len(sen.split()) <= max_len:
                pre_lst += [sen]
            else:
                tab_sen = sen.split()
                while len(tab_sen) > max_len:
                    if ';' in tab_sen[:max_len]:
                        split_id = max_len - tab_sen[:max_len][::-1].index(';')
                        pre_lst += [' '.join(tab_sen[:split_id])]
                        tab_sen = tab_sen[split_id:]
                    elif '--' in tab_sen[:max_len]:
                        split_id = max_len - tab_sen[:max_len][::-1].index('--')
                        pre_lst += [' '.join(tab_sen[:split_id])]
                        tab_sen = tab_sen[split_id:]
                    else:
                        candidates = [w.count('.') == 1 for w in tab_sen[:max_len]]
                        if sum(candidates) > 0:
                            split_id = max_len - candidates[::-1].index(True) - 1
                            a, b = tab_sen[split_id].split('.')
                            pre_lst += [' '.join(tab_sen[:split_id] + [a])]
                            tab_sen = [b] + tab_sen[split_id + 1 :]
                        else:
                            pre_lst += [' '.join(tab_sen[:max_len])]
                            tab_sen = tab_sen[max_len:]
                pre_lst += [' '.join(tab_sen)]
        res += pre_lst
    return [
        ' '.join(s.split()) for s in res if len(s.strip()) > 0
    ]  # and len(s.split()) > 1]


# pre-compute a normalized sparse tfidf vector
def tf_idf_vec_uni(sentence, vocounts, totcounts):
    sen_tab = sentence.lower().split()
    uni_dic = {}
    for _, w in enumerate(sen_tab):
        if w in vocounts:
            uni_dic[w] = -math.log(float(vocounts.get(w, 1.0)) / totcounts)
    # normalize
    uni_norm = math.sqrt(sum([x * x for x in uni_dic.values()]))
    if uni_norm > 0:
        for w in uni_dic:
            uni_dic[w] /= uni_norm
    return uni_dic


def tf_idf_vec(sentence, vocounts, totcounts):
    sen_tab = sentence.lower().split()
    uni_dic = {}
    for _, w in enumerate(sen_tab):
        uni_dic[w] = -math.log(float(vocounts.get(w, 1.0)) / totcounts)
    for i in range(len(sen_tab)):
        a = sen_tab[i]
        if i + 1 < len(sen_tab):
            b = sen_tab[i + 1]
            uni_dic[a + ' ' + b] = uni_dic[a] + uni_dic[b]
        if i + 2 < len(sen_tab):
            b = sen_tab[i + 2]
            uni_dic[a + ' ' + b] = uni_dic[a] + uni_dic[b]
    # normalize
    uni_norm = math.sqrt(sum([x * x for x in uni_dic.values()]))
    if uni_norm > 0:
        for w in uni_dic:
            uni_dic[w] /= uni_norm
    return uni_dic


# dot-product of pre-computed sparse tfidf vectors
def tf_idf_dist(dic_q, dic_t):
    dot_p = 0
    if len(dic_t) < len(dic_q):
        for w, x in dic_t.items():
            if w in dic_q:
                dot_p += x * dic_q[w]
    else:
        for w, x in dic_q.items():
            if w in dic_t:
                dot_p += x * dic_t[w]
    return dot_p


# select CommonCrawl UUIDs of desired support documents
def make_ccid_filter(ccid_maps, n_urls):
    select = {}
    for name, ccmap_ls in ccid_maps.items():
        for eli_k, cc_ls in ccmap_ls:
            for i, (cid, _) in enumerate(cc_ls[:n_urls]):
                # select[cid] = (name, eli_k[:2], eli_k[:4], eli_k, i)
                select[cid] = (name, eli_k, i)
    return select


# merge slices of collected documents
def merge_support_docs(doc_name):
    docs = []
    for f_name in glob(pjoin(doc_name, '*.json')):
        docs += json.load(open(f_name))
    print('files loaded, merging')
    merged = {}
    for eli_k, num, article in docs:
        merged[eli_k] = merged.get(eli_k, [''] * 100)
        merged[eli_k][num] = article
    print('articles merged, deduping')
    for eli_k, articles in merged.items():
        merged[eli_k] = [art for art in articles if art != '']
        merged[eli_k] = [
            x
            for i, x in enumerate(merged[eli_k])
            if (x['url'] not in [y['url'] for y in merged[eli_k][:i]])
        ]
    return list(merged.items())
