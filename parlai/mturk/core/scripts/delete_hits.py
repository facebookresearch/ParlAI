# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse
import os
import sys

import parlai.mturk.core.mturk_utils as mturk_utils

def main():
    """This script should be used after some error occurs that leaves HITs live
    while the ParlAI MTurk server down. This will search through live HITs and
    list them by task ID, letting you close down HITs that do not link to 
    any server and are thus irrecoverable.
    """
    parser = argparse.ArgumentParser(description='Delete HITs by expiring')
    parser.add_argument('--sandbox', dest='sandbox', default=False,
                        action='store_true', help='Delete HITs from sandbox')
    opt = parser.parse_args()
    sandbox = opt.sandbox
    task_group_ids = []
    group_to_hit = {}
    hits = []
    processed = 0
    found = 0
    spinner_vals = ['-','\\','|','/']
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

    print('Getting HITs from amazon MTurk server, please wait...\n')
    mturk_utils.setup_aws_credentials()
    client = mturk_utils.get_mturk_client(sandbox)
    response = client.list_hits(MaxResults=100)
    while (True):
        processed += response['NumResults']
        for hit in response['HITs']:
            if hit['NumberOfAssignmentsAvailable'] == 0:
                # Ignore hits with no assignable assignments
                continue
            if hit['HITStatus'] != 'Assignable' and \
                    hit['HITStatus'] != 'Unassignable':
                # Ignore completed hits
                continue
            question = hit['Question']
            if 'ExternalURL' in question:
                url = question.split('ExternalURL')[1]
                group_id = url.split('task_group_id=')[1]
                group_id = group_id.split('&')[0]
                group_id = group_id.split('<')[0]
                if group_id not in task_group_ids:
                    group_to_hit[group_id] = []
                    task_group_ids.append(group_id)
                group_to_hit[group_id].append(hit['HITId'])
                found += 1
        sys.stdout.write(
            '\r{} HITs processed, {} active hits found amongst {} tasks. {}   '
            .format(
                processed,
                found,
                len(task_group_ids),
                spinner_vals[((int) (processed / 100)) % 4]
            )
        )
        if 'NextToken' not in response:
            break
        response = client.list_hits(
            NextToken=response['NextToken'],
            MaxResults=100
        )

    print('\n\nTask group id - Active HITs')
    for group_id in task_group_ids:
        print('{} - {}'.format(group_id, len(group_to_hit[group_id])))

    print(
        'To clear a task, please enter the task group id of the task that you '
        'want to expire the HITs for. To exit, enter nothing'
    )

    while True:
        task_group_id = input("Enter task group id: ")
        if len(task_group_id) == 0:
            break
        elif (task_group_id not in task_group_ids):
            print('Sorry, the id you entered was not found, try again')
        else:
            num_hits = input(
                'Confirm by entering the number of HITs that will be deleted: '
            )
            if '{}'.format(len(group_to_hit[task_group_id])) == num_hits:
                hits_expired = 0
                for hit_id in group_to_hit[task_group_id]:
                    mturk_utils.expire_hit(sandbox, hit_id)
                    hits_expired += 1
                    sys.stdout.write('\rExpired hits {}'.format(hits_expired))
                print('\nAll hits for group {} have been expired.'.format(
                    task_group_id
                ))
            else:
                print(
                    'You entered {} but there are {} HITs to expire, please '
                    'try again to confirm you are ending the right task'.format(
                        num_hits,
                        len(group_to_hit[task_group_id])
                    )
                )

if __name__ == '__main__':
    main()
