#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
from os.path import join as pjoin
from os.path import isfile
from os.path import isdir
from time import time
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize, make_ccid_filter
from parlai.utils.io import PathManager

"""
Adapted from https://github.com/facebookresearch/ELI5/blob/master/data_creation/download_support_docs.py
to download specific CommonCrawl IDs and URLs.
"""


def make_docs_directory(output_dir, name):
    if not isdir(pjoin(output_dir, name)):
        subprocess.run(['mkdir', pjoin(output_dir, name)], stdout=subprocess.PIPE)
    for i in range(10):
        if not isdir(pjoin(output_dir, name, str(i))):
            subprocess.run(
                ['mkdir', pjoin(output_dir, name, str(i))], stdout=subprocess.PIPE
            )


def select_specific_ids(ccids):
    select = {}
    for i, ccid in enumerate(ccids):
        if not ccid.startswith('<urn:uuid:'):
            ccid = '<urn:uuid:' + ccid
        if not ccid.endswith('>'):
            ccid = ccid + '>'
        select[ccid] = ('specific_ids', i)
    return select


def select_specific_urls(urls):
    select = {}
    for i, url in enumerate(urls):
        if url.startswith('http://') or url.startswith('https://'):
            url = url.split('//', 1)[-1]
        select[url] = ('specific_urls', i)
    return select


def check_url(select, url):
    if url.startswith('http://') or url.startswith('https://'):
        url = url.split('//', 1)[-1]
    return select.get(url, False)


def setup_args():
    """
    Set up args.
    """
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    cc = parser.add_argument_group('Download Support Docs')
    cc.add_argument(
        '-nw',
        '--slsize',
        default=716,
        type=int,
        metavar='N',
        help='number of wet files in a slice',
    )
    cc.add_argument(
        '-ns',
        '--slnum',
        default=0,
        type=int,
        metavar='N',
        help='commoncrawl slice number [0, ..., 71520 / args.slsize]',
    )
    cc.add_argument(
        '-wf',
        '--wet_urls',
        default='pre_computed/wet.paths',
        type=str,
        help='path from data folder to file containing WET file URLs',
    )
    cc.add_argument(
        '-sr_l',
        '--subreddit_names',
        default='["explainlikeimfive"]',
        type=str,
        help='subreddit names',
    )
    cc.add_argument(
        '-nu',
        '--n_urls',
        default=100,
        type=int,
        metavar='N',
        help='number of support documents to gather for each example',
    )
    cc.add_argument(
        '-sfq',
        '--save_freq',
        default=50,
        type=int,
        metavar='N',
        help='how often are results written to file',
    )
    cc.add_argument(
        '-o',
        '--output_dir',
        default='eli5',
        type=str,
        help='where to save the output in data folder',
    )
    cc.add_argument(
        '-u',
        '--urls',
        type=str,
        help='path to a json file of URLs to gather (in a list format)',
    )
    cc.add_argument(
        '-ids',
        '--ccuids',
        type=str,
        help='path to a json file of Common Crawl IDs to gather (in a list format)',
    )
    return parser.parse_args()


def main():
    opt = setup_args()
    output_dir = pjoin(
        opt['datapath'], opt['output_dir'], 'processed_data/collected_docs'
    )
    wet_urls_path = pjoin(opt['datapath'], opt['output_dir'], opt['wet_urls'])
    f = open(wet_urls_path, buffering=4096)
    url_lst = [line.strip() for line in f if line.strip() != '']
    f.close()
    if opt['urls']:
        with PathManager.open(opt['urls']) as f:
            specific_urls = json.load(f)
        using_specific_urls = True
        using_specific_ids = False
    elif opt['ccuids']:
        with PathManager.open(opt['ccuids']) as f:
            specific_ids = json.load(f)
        using_specific_urls = False
        using_specific_ids = True
    else:
        sr_names = json.loads(opt['subreddit_names'])
        using_specific_urls = False
        using_specific_ids = False
    print("loading URL selection")
    ccrawl_ids_maps = {}
    reddit_id_group = {}
    sr_names = json.loads(opt['subreddit_names'])
    # make a list of the CommonCrawl UIDs or URLs we want to process and keep
    if using_specific_urls:
        select_urls = select_specific_urls(specific_urls)
    elif using_specific_ids:
        select_ccid = select_specific_ids(specific_ids)
    else:
        for name in sr_names:
            print(name)
            ccrawl_ids_maps[name] = json.load(
                open('pre_computed/%s_ccrawl_ids.json' % (name,))
            )
            for i, (k, _) in enumerate(ccrawl_ids_maps[name]):
                reddit_id_group[k] = (i * 10) // len(ccrawl_ids_maps[name])
        select_ccid = make_ccid_filter(ccrawl_ids_maps, opt['n_urls'])

    print("loaded URL selection")
    # organize directories
    if not isdir(output_dir):
        subprocess.run(['mkdir', output_dir], stdout=subprocess.PIPE)
        if not isdir(pjoin(output_dir, 'tmp')):
            subprocess.run(['mkdir', pjoin(output_dir, 'tmp')], stdout=subprocess.PIPE)
    if using_specific_ids:
        make_docs_directory(output_dir, 'specific_ids')
    elif using_specific_urls:
        make_docs_directory(output_dir, 'specific_urls')
    else:
        for name in sr_names:
            make_docs_directory(output_dir, name)

    # check whether some ccrawl files have already been processed for this slice
    if using_specific_ids:
        articles = dict([('specific_ids', dict([(i, []) for i in range(10)]))])
        mode = 'ids'
    elif using_specific_urls:
        articles = dict([('specific_urls', dict([(i, []) for i in range(10)]))])
        mode = 'urls'
    else:
        articles = dict(
            [(name, dict([(i, []) for i in range(10)])) for name in sr_names]
        )
        mode = 'subreddits'
    # check progress of slice or if slice is finished
    if isfile(pjoin(output_dir, 'tmp', 'counts_%s_%d.json' % (mode, opt['slnum']))):
        start_line = json.load(
            open(pjoin(output_dir, 'tmp', 'counts_%s_%d.json' % (mode, opt['slnum'])))
        )
        if start_line == 'finished':
            return True
        for name in sr_names:
            for i_st in range(10):
                d_name = pjoin(output_dir, name, str(i_st))
                articles[name][i] = json.load(
                    open(pjoin(d_name, "docs_slice_%05d.json" % (opt['slnum'])))
                )
        print(
            "loaded previously downloaded pages:",
            start_line - opt['slnum'] * opt['slsize'],
        )
    else:
        start_line = opt['slnum'] * opt['slsize']
    # Download and parse slice of args.slsize WET files
    st_time = time()
    for i in range(start_line, min((opt['slnum'] + 1) * opt['slsize'], len(url_lst))):
        # Download wet file from amazon AWS
        dl_time = time()
        fname = url_lst[i].split('/')[-1][:-3]
        # download and unzip if necessary
        fpath = pjoin(output_dir, 'tmp', fname)
        print("processing", fpath)
        if not isfile(fpath):
            ct_try = 0
            while not isfile(fpath):
                subprocess.run(['rm', fpath + ".gz"], stdout=subprocess.PIPE)
                while not isfile(fpath + ".gz"):
                    url = "https://commoncrawl.s3.amazonaws.com/" + url_lst[i]
                    subprocess.run(
                        ['wget', '-P', pjoin(output_dir, 'tmp'), url],
                        stdout=subprocess.PIPE,
                    )
                    print("download:", time() - dl_time)
                    ct_try += 1
                    if ct_try > 5 and not isfile(fpath + ".gz"):
                        print("giving up on file", fname)
                        break
                downloaded = isfile(fpath + ".gz")
                if downloaded:
                    subprocess.run(['gunzip', fpath + ".gz"], stdout=subprocess.PIPE)
                    print("download and gunzip:", time() - dl_time)
                if ct_try > 5 and not isfile(fpath):
                    print("giving up on file", fname)
                    break
        else:
            downloaded = isfile(fpath)
        if not downloaded:
            print("FAILED DOWNLOADING ", fpath)
            continue
        # Extract, tokenize, and filter articles by language
        f = open(fpath, buffering=4096)
        article_url = ''
        article_id = ''
        article_txt = ''
        last_line = ''
        read_text = False
        ct = 0
        start_time = time()
        ccid_path_tuple = False
        # check and save pages by IDs if getting posts by IDs, or by URLs
        # if using URLs
        for line in f:
            if line.startswith("WARC/1.0"):
                if ccid_path_tuple:
                    ct += 1
                    article = {
                        'ccid': article_id,
                        'url': article_url,
                        'text': word_url_tokenize(article_txt),
                    }
                    if not using_specific_urls and not using_specific_ids:
                        name, eli_k, num = ccid_path_tuple
                        articles[name][reddit_id_group[eli_k]] += [
                            (eli_k, num, article)
                        ]
                    else:
                        name, num = ccid_path_tuple
                        articles[name][num % 10] += [(num, article)]
                article_txt = ''
                read_text = False
            if line.startswith("WARC-Target-URI"):
                try:
                    article_url = line.strip().split()[-1]
                    if using_specific_urls:
                        ccid_path_tuple = check_url(select_urls, article_url)
                except Exception:
                    article_url = '<UNK>'
                    if using_specific_urls:
                        ccid_path_tuple = False
            if line.startswith("WARC-Record-ID"):
                try:
                    article_id = line.strip().split()[-1]
                    if not using_specific_urls:
                        ccid_path_tuple = select_ccid.get(article_id, False)
                except Exception:
                    article_id = '<UNK>'
                    if not using_specific_urls:
                        ccid_path_tuple = False
            if read_text and (last_line.strip() + line.strip()) != '':
                article_txt += line + '\n'
                last_line = line
            if line.startswith("Content-Length: ") and ccid_path_tuple:
                read_text = True
        if ccid_path_tuple:
            ct += 1
            article = {
                'ccid': article_id,
                'url': article_url,
                'text': word_url_tokenize(article_txt),
            }

            if not using_specific_urls and not using_specific_ids:
                name, eli_k, num = ccid_path_tuple
                articles[name][reddit_id_group[eli_k]] += [(eli_k, num, article)]
            else:
                name, num = ccid_path_tuple
                articles[name][num % 10] += [(num, article)]
        f.close()
        subprocess.run(['rm', fpath], stdout=subprocess.PIPE)
        # periodically save slice
        print(">>>>>>>>>> ARTICLES FOUND %d in %.2f" % (ct, time() - start_time))
        if i % opt['save_freq'] == opt['save_freq'] - 1:
            for name, elik_maps in articles.items():
                print('saving', name, i, len(elik_maps))
                for i_st, ls in elik_maps.items():
                    d_name = pjoin(output_dir, name, str(i_st))
                    if not isdir(d_name):
                        subprocess.run(['mkdir', d_name], stdout=subprocess.PIPE)
                    json.dump(
                        ls,
                        open(
                            pjoin(d_name, "docs_slice_%05d.json" % (opt['slnum'])), 'w'
                        ),
                    )
            json.dump(
                i + 1,
                open(
                    pjoin(
                        output_dir, 'tmp', 'counts_%s_%d.json' % (mode, opt['slnum'])
                    ),
                    'w',
                ),
            )
            print('saved json files %.2f' % (time() - start_time,))
        subprocess.run(['rm', fpath], stdout=subprocess.PIPE)
    # save items to slices
    for name, elik_maps in articles.items():
        print('saving', name, i, len(elik_maps))
        for i_st, ls in elik_maps.items():
            d_name = pjoin(output_dir, name, str(i_st))
            if not isdir(d_name):
                subprocess.run(['mkdir', d_name], stdout=subprocess.PIPE)
            json.dump(
                ls, open(pjoin(d_name, "docs_slice_%05d.json" % (opt['slnum'])), 'w')
            )
    print('saved json files %.2f' % (time() - start_time,))
    json.dump(
        'finished',
        open(pjoin(output_dir, 'tmp', 'counts_%s_%d.json' % (mode, opt['slnum'])), 'w'),
    )
    print("processing slice %d took %f seconds" % (i, time() - st_time))


if __name__ == '__main__':
    main()
