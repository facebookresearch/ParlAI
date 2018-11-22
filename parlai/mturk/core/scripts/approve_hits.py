import argparse
import parlai.mturk.core.mturk_utils as mturk_utils


def main():
    """This script should be used to approve HITs when auto-approve will not suffice."""
    parser = argparse.ArgumentParser(description='Approve HITs directly')
    parser.add_argument('--sandbox', dest='sandbox', default=False,
                        action='store_true', help='Test in sandbox')
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

    while True:
        hit_id = input("Enter HIT id: ")
        if len(hit_id) == 0:
            break
        # MTurkManager.get_assignments_for_hist()
        assignments_info = client.list_assignments_for_hit(HITId=hit_id)
        assignments = assignments_info.get('Assignments', [])
        # MTurkManager.approve_assignments_for_hit()
        for assignment in assignments:
            assignment_id = assignment['AssignmentId']
            resp = client.approve_assignment(AssignmentId=assignment_id,
                                             OverrideRejection=False)
            print(resp)


if __name__ == '__main__':
    main()
