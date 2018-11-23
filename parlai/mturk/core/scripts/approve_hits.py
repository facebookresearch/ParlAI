import argparse
import numpy as np
import parlai.mturk.core.mturk_utils as mturk_utils


chunks = [
    0,
    905,
    1390,
    9872,
    251,
    669,
    651,
    544,
    456,
    44,
    481,
    672,
    595,
    693,
    322,
    81,
    330,
    366,
    90,
    143,
    185,
    93,
    33,
    30,
    30,
    13,
    1342,
    195,
    25,
    130,
    40,
    21,
    137,
    3,
    25,
    5,
    5,
]
cum_count = np.cumsum(chunks)
# cum_count:
# array([    0,   905,  2295, 12167, 12418, 13087, 13738, 14282, 14738,
#        14782, 15263, 15935, 16530, 17223, 17545, 17626, 17956, 18322,
#        18412, 18555, 18740, 18833, 18866, 18896, 18926, 18939, 20281,
#        20476, 20501, 20631, 20671, 20692, 20829, 20832, 20857, 20862,
#        20867])


def main():
    """This script should be used to approve HITs when auto-approve will not suffice."""
    parser = argparse.ArgumentParser(description='Approve HITs directly')
    parser.add_argument('--sandbox', dest='sandbox', default=False,
                        action='store_true', help='Test in sandbox')
    parser.add_argument('-if', '--infile', dest='infile', default='hit_ids.txt')
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

    # For interactive mode
    # while True:
    #     hit_id = input("Enter HIT id: ")
    #     if len(hit_id) == 0:
    #         break

    # Read from file
    with open(opt.infile, 'r') as f:
        hit_ids = f.readlines()

    # Write explicitly
    # hit_ids = [
        # FORMER
        # '33BFF6QPI1YFX6KNNL370WSI8463W3',
        # '3GS542CVJVA7ZPUJ8TH819IRO3E95J',
        # '37M4O367VJ5M69DDX2LD6VOBO2JM5T',
        # '3Y7LTZE0YT93QV2BDSJFM9C02JIZUL',
        # '3CRWSLD91KR8EJJV2HHCF1O25XQMO0',
        # '39RRBHZ0AUO33ARBIITKIVEJG3BVZG',
        # '37VHPF5VYCQSK5KOIBY2FJTQWDHC82',
        # '3VJ4PFXFJ3U2PNU3103G5GLRTYBUAF',
        # '3Y7LTZE0YT93QV2BDSJFM9C02JIZUL',
        # '35NNO802AVJ40FTSGNJ78JHGYCWINS',
        # '3YO4AH2FPD7EWHP4SPMI69CPHV80QC',
        # '371Q3BEXDHWNBIA7ONOF78UCHPBZSP',
        # '3Y7LTZE0YT93QV2BDSJFM9C02JIZUL',
        # '3M93N4X8HKAXB35361LTJE6MYS4JSW',
        # '3OLZC0DJ8J2H8K21IE5YI0BTKHAIVP',
        # '39N6W9XWRDAKGNRBX1SF4N27NLBGYV',
        # '3YO4AH2FPD7EWHP4SPMI69CPHV80QC',
        # '371Q3BEXDHWNBIA7ONOF78UCHPBZSP',
        # '3M93N4X8HKAXB35361LTJE6MYS4JSW',
    # ]

    success = 0
    suc_seq = 0
    failure = 0
    fail_seq = 0
    chunk = 0
    i = -1
    while True:
        i += 1
        # Fail too many times, jump to next chunk
        if fail_seq >= 20:
            chunk += 1
            suc_seq = 0
            fail_seq = 0
            i = cum_count[chunk]
            print(f"Jump to chunk {chunk} at index {i}")
        # Keep track of what chunk we're in
        if chunk < len(chunks) and i >= cum_count[chunk + 1]:
            suc_seq = 0
            fail_seq = 0
            chunk += 1
            print(f"Advancing to chunk {chunk}")

        hit_id = hit_ids[i].strip()
        # MTurkManager.get_assignments_for_hist()
        assignments_info = client.list_assignments_for_hit(HITId=hit_id)
        assignments = assignments_info.get('Assignments', [])
        # MTurkManager.approve_assignments_for_hit()
        if len(assignments) == 0:
            print(f"Empty on index {i}")
        for assignment in assignments:
            assignment_id = assignment['AssignmentId']
            try:
                resp = client.approve_assignment(AssignmentId=assignment_id,
                                                 OverrideRejection=True)
                success += 1
                suc_seq += 1
                fail_seq = 0
                print(f"Success on index {i} ({suc_seq} in a row)")
            except BaseException as e:
                failure += 1
                fail_seq += 1
                suc_seq = 0
                print(f"Failed on index {i} ({fail_seq} in a row)")


if __name__ == '__main__':
    main()
