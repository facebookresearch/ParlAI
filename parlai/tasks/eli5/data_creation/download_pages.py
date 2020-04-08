# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import subprocess
import os

from os.path import join as pjoin
from os.path import isfile
from os.path import isdir
from subprocess import check_output
from time import time

from data_utils import *


def main():
    parser = argparse.ArgumentParser(
        description='select support document pages from common crawl'
    )
    parser.add_argument(
        '-nw',
        '--slsize',
        default=716,
        type=int,
        metavar='N',
        help='number of wet files in a slice',
    )
    parser.add_argument(
        '-ns',
        '--slnum',
        default=0,
        type=int,
        metavar='N',
        help='commoncrawl slice number [0, ..., 71520 / args.slsize]',
    )
    parser.add_argument(
        '-wf',
        '--wet_urls',
        default='pre_computed/wet.paths',
        type=str,
        help='path to file containing WET file URLs',
    )
    parser.add_argument(
        '-sr_l',
        '--subreddit_names',
        default='["explainlikeimfive"]',
        type=str,
        help='subreddit names',
    )
    parser.add_argument(
        '-nu',
        '--n_urls',
        default=100,
        type=int,
        metavar='N',
        help='number of support documents to gather for each example',
    )
    parser.add_argument(
        '-sfq',
        '--save_freq',
        default=50,
        type=int,
        metavar='N',
        help='how often are results written to file',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        default='processed_data/collected_docs',
        type=str,
        help='where to save the output',
    )
    # mine args
    parser.add_argument(
        '-u', '--urls', default='[]', type=str, help='Gather pages by URL'
    )
    parser.add_argument(
        '-ids', '--ccuids', default='[]', type=str, help='Gather pages by CCUID'
    )
    args = parser.parse_args()
    # parse full list of wet urls
    # slice urls for WET files can be found at https://commoncrawl.org/2018/08/august-2018-crawl-archive-now-available/
    # $ wget https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-34/wet.paths.gz
    # $ gunzip wet.paths.gz

    # url example
    #"http://098-ckr.com/test/read.php/delete/1355699330/"
    specific_urls = json.loads(args.urls)
    if not specific_urls:
        f = open(args.wet_urls, buffering=4096)
        url_lst = [line.strip() for line in f if line.strip() != '']
        f.close()
        sr_names = json.loads(args.subreddit_names)
        using_specific_urls = False
    else:
        f = open(args.wet_urls, buffering=4096)
        url_lst = [line.strip() for line in f if line.strip() != '']
        f.close()
        specific_urls = [line.strip() for line in specific_urls if line.strip() != '']
        using_specific_urls = True
        
    print("loading URL selection")
    ccrawl_ids_maps = {}
    reddit_id_group = {}

    # mine variables
    ccuids = json.loads(args.ccuids)

    # make a list of the CommonCrawl UIDs we want to process and keep
    print('MAKE FILTER')
    if not using_specific_urls:
        select_ccid = make_ccid_filter(ccrawl_ids_maps, args.n_urls)
    else:
        select_urls = select_specific_urls(specific_urls)

    # eli_k = post id corresponding to eli5 post
    for i, ccid in enumerate(ccuids):
        select_ccid[ccid] = ('specific_items', i)
    print("loaded URL selection")
    # organize directories
    # make tmp directory
    if not isdir(args.output_dir):
        subprocess.run(['mkdir', args.output_dir], stdout=subprocess.PIPE)
        if not isdir(pjoin(args.output_dir, 'tmp')):
            subprocess.run(
                ['mkdir', pjoin(args.output_dir, 'tmp')], stdout=subprocess.PIPE
            )
    # make specific url/ids directory
    if not isdir(pjoin(args.output_dir, 'specific_items')):
        subprocess.run(
            ['mkdir', pjoin(args.output_dir, 'specific_items')], stdout=subprocess.PIPE
        )
    for i in range(10):
        if not isdir(pjoin(args.output_dir, 'specific_items', str(i))):
            subprocess.run(
                ['mkdir', pjoin(args.output_dir, 'specific_items', str(i))],
                stdout=subprocess.PIPE,
            )
    # check whether some ccrawl files have already been processed for this slice
    articles = dict([('specific_items', dict([(i, []) for i in range(10)]))])
    if isfile(pjoin(args.output_dir, 'tmp', 'counts_%d.json' % (args.slnum))):
        start_line = json.load(
            open(pjoin(args.output_dir, 'tmp', 'counts_%d.json' % (args.slnum)))
        )
        if start_line == 'finished':
            return True
        
        # check in specific items
        for i_st in range(10):
            d_name = pjoin(args.output_dir, 'specific_items', str(i_st))
            articles['specific_items'][i] = json.load(
                open(pjoin(d_name, "docs_slice_%05d.json" % (args.slnum)))
            )
        print(
            "loaded previously downloaded pages:", start_line - args.slnum * args.slsize
        )
    else:
        start_line = args.slnum * args.slsize
    # Download and parse slice of args.slsize WET files
    st_time = time()
    for i in range(start_line, min((args.slnum + 1) * args.slsize, len(url_lst))):
        # Download wet file from amazon AWS
        dl_time = time()
        fname = url_lst[i].split('/')[-1][:-3]
        # download and unzip if necessary
        fpath = pjoin(args.output_dir, 'tmp', fname)
        print("processing", fpath)
        if not isfile(fpath):
            ct_try = 0
            while not isfile(fpath):
                resp_c = subprocess.run(['rm', fpath + ".gz"], stdout=subprocess.PIPE)
                while not isfile(fpath + ".gz"):
                    # example url:
                    #  https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-34/segments/1534221208676.20/wet/CC-MAIN-20180814062251-20180814082251-00002.warc.wet.gz
                    url = "https://commoncrawl.s3.amazonaws.com/" + url_lst[i]
                    resp_a = subprocess.run(
                        ['wget', '-P', pjoin(args.output_dir, 'tmp'), url],
                        stdout=subprocess.PIPE,
                    )
                    print("download:", time() - dl_time)
                    ct_try += 1
                    if ct_try > 5 and not isfile(fpath + ".gz"):
                        print("giving up on file", fname)
                        break
                downloaded = isfile(fpath + ".gz")
                if downloaded:
                    resp_b = subprocess.run(
                        ['gunzip', fpath + ".gz"], stdout=subprocess.PIPE
                    )
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
        for line in f:
            if line.startswith("WARC/1.0"):
                if ccid_path_tuple:
                    ct += 1
                    article = {
                        'ccid': article_id,
                        'url': article_url,
                        'text': word_url_tokenize(article_txt),
                        # 'text': 'test',
                    }
                    name, num = ccid_path_tuple
                    articles[name][num % 10] += [(num, article)]
                article_txt = ''
                read_text = False
            if line.startswith("WARC-Target-URI"):
                try:
                    article_url = line.strip().split()[-1]
                    if using_specific_urls:
                        ccid_path_tuple = check_url(select_urls, article_url)
                except:
                    article_url = '<UNK>'
                    ccid_path_tuple = False
            if line.startswith("WARC-Record-ID"):
                try:
                    article_id = line.strip().split()[-1]
                    if not using_specific_urls: 
                        ccid_path_tuple = select_ccid.get(article_id, False)
                except:
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
                # 'text': 'test',
            }
            name, eli_k, num = ccid_path_tuple
            # articles pertaining to a specific reddit post
            if using_specific_urls:
                # name, eli_k, num = 'specific_urls', None, ct-1
                articles[name][num % 10] += [(num, article)]
            else:
                articles[name][reddit_id_group[eli_k]] += [(eli_k, num, article)]
        f.close()
        resp_c = subprocess.run(['rm', fpath], stdout=subprocess.PIPE)
        print(">>>>>>>>>> ARTICLES FOUND %d in %.2f" % (ct, time() - start_time))
        if i % args.save_freq == args.save_freq - 1:
            for name, elik_maps in articles.items():
                print('saving', name, i, len(elik_maps))
                for i_st, ls in elik_maps.items():
                    d_name = pjoin(args.output_dir, name, str(i_st))
                    if not isdir(d_name):
                        subprocess.run(['mkdir', d_name], stdout=subprocess.PIPE)
                    json.dump(
                        ls,
                        open(pjoin(d_name, "docs_slice_%05d.json" % (args.slnum)), 'w'),
                    )
            json.dump(
                i + 1,
                open(
                    pjoin(args.output_dir, 'tmp', 'counts_%d.json' % (args.slnum)), 'w'
                ),
            )
            print('saved json files %.2f' % (time() - start_time,))
        resp_c = subprocess.run(['rm', fpath], stdout=subprocess.PIPE)
    print('final save')
    for name, elik_maps in articles.items():
        print('saving', name, i, len(elik_maps))
        for i_st, ls in elik_maps.items():
            d_name = pjoin(args.output_dir, name, str(i_st))
            if not isdir(d_name):
                subprocess.run(['mkdir', d_name], stdout=subprocess.PIPE)
            json.dump(
                ls, open(pjoin(d_name, "docs_slice_%05d.json" % (args.slnum)), 'w')
            )
    print('saved json files %.2f' % (time() - start_time,))
    json.dump(
        'finished',
        open(pjoin(args.output_dir, 'tmp', 'counts_%d.json' % (args.slnum)), 'w'),
    )
    print("processing slice %d took %f seconds" % (i, time() - st_time))


if __name__ == '__main__':
    main()
