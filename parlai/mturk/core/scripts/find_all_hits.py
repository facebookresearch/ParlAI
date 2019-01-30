#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse
import sys

import parlai.mturk.core.mturk_utils as mturk_utils


def main():
    """This script should be used to approve HITs when auto-approve will not suffice."""
    parser = argparse.ArgumentParser(description='Approve HITs directly')
    parser.add_argument('--sandbox', dest='sandbox', default=False,
                        action='store_true', help='Test in sandbox')
    parser.add_argument('-of', '--outfile', dest='outfile', default='hit_ids.txt')
    opt = parser.parse_args()
    sandbox = opt.sandbox

    if sandbox:
        print(
            'About to connect to the SANDBOX server. These payments will not '
            'actually be paid out.'
        )
    else:
        print(
            'About to connect to the LIVE server. These payments will be '
            'deducted from your account balance.'
        )

    mturk_utils.setup_aws_credentials()
    client = mturk_utils.get_mturk_client(sandbox)

    task_group_ids = []
    group_to_hit = {}
    processed = 0
    found = 0
    spinner_vals = ['-', '\\', '|', '/']
    if sandbox:
        print(
            'About to query the SANDBOX server, these HITs will be active HITs'
            ' from within the MTurk requester sandbox'
        )
    else:
        print(
            'About to query the LIVE server, these HITs will be active HITs '
            'potentially being worked on by real Turkers right now'
        )

    print('Getting HITs from amazon MTurk server, please wait...\n'
          'or use CTRL-C to skip to approving some of what is found.\n')
    mturk_utils.setup_aws_credentials()
    client = mturk_utils.get_mturk_client(sandbox)
    response = client.list_hits(MaxResults=100)
    try:
        while (True):
            processed += response['NumResults']
            for hit in response['HITs']:
                question = hit['Question']
                try:
                    if 'ExternalURL' in question:
                        url = question.split('ExternalURL')[1]
                        group_id = url.split('task_group_id=')[1]
                        group_id = group_id.split('&')[0]
                        group_id = group_id.split('<')[0]
                        if group_id not in task_group_ids:
                            sys.stdout.write(
                                '\rFound group {}                         '
                                '                                         '
                                '\n'.format(group_id)
                            )
                            group_to_hit[group_id] = {}
                            task_group_ids.append(group_id)
                        group_to_hit[group_id][hit['HITId']] = hit['HITStatus']
                        found += 1
                except IndexError:
                    pass  # This wasn't the right HIT

            sys.stdout.write(
                '\r{} HITs processed, {} active hits'
                ' found amongst {} tasks. {}        '
                .format(
                    processed,
                    found,
                    len(task_group_ids),
                    spinner_vals[((int)(processed / 100)) % 4]
                )
            )
            if 'NextToken' not in response:
                break
            response = client.list_hits(
                NextToken=response['NextToken'],
                MaxResults=100
            )

    except BaseException as e:
        print(e)
        pass

    print('\n\nTask group id - Active HITs')
    for group_id in task_group_ids:
        print('{} - {}'.format(group_id, len(group_to_hit[group_id])))

    print(
        'To clear a task, please enter the task group id of the task that you '
        'want to expire the HITs for. To exit, enter nothing'
    )

    hit_ids = []
    for task_group_id in task_group_ids:
        if ('rating' in task_group_id or 'explanation' in task_group_id or
                'metadialog' in task_group_id):
            try:
                for hit_id, status in group_to_hit[task_group_id].items():
                    hit_ids.append(hit_id)
            except BaseException as e:
                print(e)
                print(f"Failed to collect hit_ids for group {task_group_id}.")
    print('\nTotal hit_ids collected:{}.'.format(len(hit_ids)))

    with open(opt.outfile, 'w') as f:
        for hit_id in hit_ids:
            f.write("%s\n" % hit_id)


if __name__ == '__main__':
    main()
