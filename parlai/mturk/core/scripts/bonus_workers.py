#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import parlai.mturk.core.mturk_utils as mturk_utils


def main():
    """This script should be used to compensate workers that have not recieved
    proper payment for the completion of tasks due to issues on our end.
    It's important to make sure you keep a requester reputation up.
    """
    parser = argparse.ArgumentParser(description='Bonus workers directly')
    parser.add_argument('--sandbox', dest='sandbox', default=False,
                        action='store_true', help='Test bonus in sandbox')
    parser.add_argument('--hit-id', dest='use_hit_id', default=False,
                        action='store_true',
                        help='Use HIT id instead of assignment id')
    opt = parser.parse_args()
    sandbox = opt.sandbox

    if sandbox:
        print(
            'About to connect to the SANDBOX server. These bonuses will not '
            'actually be paid out.'
        )
    else:
        print(
            'About to connect to the LIVE server. These bonuses will be '
            'deducted from your account balance.'
        )

    mturk_utils.setup_aws_credentials()
    client = mturk_utils.get_mturk_client(sandbox)

    while True:
        worker_id = input("Enter worker id: ")
        if len(worker_id) == 0:
            break
        bonus_amount = input("Enter bonus amount: ")
        if opt.use_hit_id:
            hit_id = input("Enter HIT id: ")
            listed = client.list_assignments_for_hit(HITId=hit_id)
            assignment_id = listed['Assignments'][0]['AssignmentId']
        else:
            assignment_id = input("Enter assignment id: ")
        reason = input("Enter reason: ")
        input(
            "Press enter to bonus {} to worker {} for reason '{}' on "
            "assignment {}.".format(bonus_amount, worker_id,
                                    reason, assignment_id)
        )
        resp = client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=str(bonus_amount),
            AssignmentId=assignment_id,
            Reason=reason
        )
        print(resp)


if __name__ == '__main__':
    main()
