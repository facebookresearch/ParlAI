#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import os
import time
import uuid
from datetime import datetime

from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.agents import AssignState

import parlai.mturk.core.mturk_data_handler as DataHandlerFile

data_dir = os.path.dirname(os.path.abspath(__file__))
DataHandlerFile.data_dir = data_dir


class TestDataHandler(unittest.TestCase):
    '''Various unit tests for the SQLite database'''

    DB_NAME = 'test_db.db'

    def setUp(self):
        if os.path.exists(os.path.join(data_dir, self.DB_NAME)):
            os.remove(os.path.join(data_dir, self.DB_NAME))

    def tearDown(self):
        if os.path.exists(os.path.join(data_dir, self.DB_NAME)):
            os.remove(os.path.join(data_dir, self.DB_NAME))

    def create_hit(self):
        return {
            'HIT': {
                'HITId': str(uuid.uuid4()),
                'Expiration': datetime.today().replace(microsecond=0),
                'HITStatus': str(uuid.uuid4()),
                'NumberOfAssignmentsPending': 1,
                'NumberOfAssignmentsAvailable': 2,
                'NumberOfAssignmentsCompleted': 3,
            }
        }

    def assertHITEqual(self, mturk_hit, db_hit, run_id):
        self.assertEqual(mturk_hit['HIT']['HITId'], db_hit['hit_id'])
        self.assertEqual(mturk_hit['HIT']['Expiration'],
                         datetime.fromtimestamp(db_hit['expiration']))
        self.assertEqual(mturk_hit['HIT']['HITStatus'], db_hit['hit_status'])
        self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsPending'],
                         db_hit['assignments_pending'])
        self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsAvailable'],
                         db_hit['assignments_available'])
        self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsCompleted'],
                         db_hit['assignments_complete'])
        self.assertEqual(run_id, db_hit['run_id'])

    def test_init_db(self):
        db_logger = MTurkDataHandler('test1', file_name=self.DB_NAME)
        conn = db_logger._get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM runs;')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(*) FROM hits;')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(*) FROM assignments;')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(*) FROM workers;')
        self.assertEqual(c.fetchone()[0], 0)
        c.execute('SELECT COUNT(*) FROM pairings;')
        self.assertEqual(c.fetchone()[0], 0)

    def test_create_get_run(self):
        run_id = 'Test_run_1'
        hits_created = 10
        db_logger = MTurkDataHandler('test2', file_name=self.DB_NAME)

        # Ensure a run logged can be retrieved
        db_logger.log_new_run(hits_created, 'testing', run_id)
        run_data = db_logger.get_run_data(run_id)
        self.assertEqual(run_data['run_id'], run_id)
        self.assertEqual(run_data['created'], 0)
        self.assertEqual(run_data['completed'], 0)
        self.assertEqual(run_data['maximum'], hits_created)
        self.assertEqual(run_data['failed'], 0)

        # Assert missed entries are None
        self.assertIsNone(db_logger.get_run_data('fake_id'))

    def test_create_update_hits(self):
        run_id = 'Test_run_2'
        hits_created = 10
        db_logger = MTurkDataHandler(file_name=self.DB_NAME)
        db_logger.log_new_run(hits_created, 'testing', run_id)
        HIT1 = self.create_hit()
        HIT2 = self.create_hit()
        HIT3 = self.create_hit()

        # Ensure logging without group id fails
        with self.assertRaises(AssertionError):
            db_logger.log_hit_status(HIT1)

        # Log created hits through one logger
        db_logger.log_hit_status(HIT1, run_id)
        db_logger.log_hit_status(HIT2, run_id)

        # Create new handler, this one with the group id created, ensure
        # the log works fine
        db_logger = MTurkDataHandler(run_id, file_name=self.DB_NAME)
        db_logger.log_hit_status(HIT3)

        # Ensure all of the expected hits are there
        run_data = db_logger.get_run_data(run_id)
        self.assertEqual(run_data['run_id'], run_id)
        self.assertEqual(run_data['created'], 3)
        self.assertEqual(run_data['completed'], 0)
        self.assertEqual(run_data['maximum'], hits_created)
        self.assertEqual(run_data['failed'], 0)

        # Ensure the hit details are correct
        for hit in [HIT1, HIT2, HIT3]:
            hit_db_data = db_logger.get_hit_data(hit['HIT']['HITId'])
            self.assertHITEqual(hit, hit_db_data, run_id)

        # Update the data on a HIT, ensure that the run data stays the same
        # but the HIT data updates
        test_status = 'TEST_STATUS'
        HIT2['HIT']['HITStatus'] = test_status
        db_logger.log_hit_status(HIT2)

        # Ensure all of the expected hits are there
        run_data = db_logger.get_run_data(run_id)
        self.assertEqual(run_data['run_id'], run_id)
        self.assertEqual(run_data['created'], 3)
        self.assertEqual(run_data['completed'], 0)
        self.assertEqual(run_data['maximum'], hits_created)
        self.assertEqual(run_data['failed'], 0)

        # Ensure the hit details are correct
        for hit in [HIT1, HIT2, HIT3]:
            hit_db_data = db_logger.get_hit_data(hit['HIT']['HITId'])
            self.assertHITEqual(hit, hit_db_data, run_id)

        # Ensure requesting a hit that doesn't exist returns none
        self.assertIsNone(db_logger.get_hit_data('fake_id'))

    def test_worker_workflows(self):
        run_id = 'Test_run_3'
        hits_created = 10
        db_logger = MTurkDataHandler(run_id, file_name=self.DB_NAME)
        db_logger.log_new_run(hits_created, 'testing', run_id)
        HIT1 = self.create_hit()
        HIT2 = self.create_hit()
        HIT3 = self.create_hit()
        db_logger.log_hit_status(HIT1)
        db_logger.log_hit_status(HIT2)
        db_logger.log_hit_status(HIT3)

        worker_id_1 = 'TEST_WORKER_ID_1'
        worker_id_2 = 'TEST_WORKER_ID_2'
        assignment_id_1 = 'TEST_ASSIGNMENT_ID_1'
        assignment_id_2 = 'TEST_ASSIGNMENT_ID_2'
        assignment_id_3 = 'TEST_ASSIGNMENT_ID_3'

        # Create two workers and assign the 3 assignments to them
        db_logger.log_worker_accept_assignment(
            worker_id_1, assignment_id_1, HIT1['HIT']['HITId'])
        db_logger.log_worker_accept_assignment(
            worker_id_2, assignment_id_2, HIT2['HIT']['HITId'])
        db_logger.log_worker_accept_assignment(
            worker_id_2, assignment_id_3, HIT3['HIT']['HITId'])

        # Ensure two workers have been created
        conn = db_logger._get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM workers;')
        self.assertEqual(c.fetchone()[0], 2)

        # Ensure non-existent worker is None
        self.assertIsNone(db_logger.get_worker_data('fake_id'))

        # Ensure the two workers have the correct expected values
        worker_1_data = db_logger.get_worker_data(worker_id_1)
        worker_2_data = db_logger.get_worker_data(worker_id_2)
        self.assertEqual(worker_1_data['worker_id'], worker_id_1)
        self.assertEqual(worker_1_data['accepted'], 1)
        self.assertEqual(worker_1_data['disconnected'], 0)
        self.assertEqual(worker_1_data['completed'], 0)
        self.assertEqual(worker_1_data['approved'], 0)
        self.assertEqual(worker_1_data['rejected'], 0)
        self.assertEqual(worker_2_data['worker_id'], worker_id_2)
        self.assertEqual(worker_2_data['accepted'], 2)
        self.assertEqual(worker_2_data['disconnected'], 0)
        self.assertEqual(worker_2_data['completed'], 0)
        self.assertEqual(worker_2_data['approved'], 0)
        self.assertEqual(worker_2_data['rejected'], 0)

        # Ensure all the assignments are marked as accepted
        c.execute('SELECT COUNT(*) FROM assignments WHERE status = ?;',
                  ('Accepted', ))
        self.assertEqual(c.fetchone()[0], 3)

        # Ensure non-existing assign is None
        self.assertIsNone(db_logger.get_assignment_data('fake_id'))

        # Check each of the assignments
        assignment_1_data = db_logger.get_assignment_data(assignment_id_1)
        assignment_2_data = db_logger.get_assignment_data(assignment_id_2)
        assignment_3_data = db_logger.get_assignment_data(assignment_id_3)
        self.assertEqual(assignment_1_data['assignment_id'], assignment_id_1)
        self.assertEqual(assignment_1_data['status'], 'Accepted')
        self.assertEqual(assignment_1_data['approve_time'], None)
        self.assertEqual(assignment_1_data['worker_id'], worker_id_1)
        self.assertEqual(assignment_1_data['hit_id'], HIT1['HIT']['HITId'])
        self.assertEqual(assignment_2_data['assignment_id'], assignment_id_2)
        self.assertEqual(assignment_2_data['status'], 'Accepted')
        self.assertEqual(assignment_2_data['approve_time'], None)
        self.assertEqual(assignment_2_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_2_data['hit_id'], HIT2['HIT']['HITId'])
        self.assertEqual(assignment_3_data['assignment_id'], assignment_id_3)
        self.assertEqual(assignment_3_data['status'], 'Accepted')
        self.assertEqual(assignment_3_data['approve_time'], None)
        self.assertEqual(assignment_3_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_3_data['hit_id'], HIT3['HIT']['HITId'])

        # Ensure three pairings have been created, one for each assignment
        c.execute('SELECT COUNT(*) FROM pairings')
        self.assertEqual(c.fetchone()[0], 3)

        # Ensure pairings are accurate
        self.assertIsNone(db_logger.get_worker_assignment_pairing(
            worker_id_1, assignment_id_3))

        pair_1 = db_logger.get_worker_assignment_pairing(
            worker_id_1, assignment_id_1)
        pair_2 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_2)
        pair_3 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_3)
        for f in ['onboarding_start', 'onboarding_end', 'task_start',
                  'task_end', 'conversation_id']:
            for pair in [pair_1, pair_2, pair_3]:
                self.assertIsNone(pair[f])
        self.assertEqual(pair_1['status'], AssignState.STATUS_NONE)
        self.assertEqual(pair_2['status'], AssignState.STATUS_NONE)
        self.assertEqual(pair_3['status'], AssignState.STATUS_NONE)
        self.assertEqual(pair_1['worker_id'], worker_id_1)
        self.assertEqual(pair_2['worker_id'], worker_id_2)
        self.assertEqual(pair_3['worker_id'], worker_id_2)
        self.assertEqual(pair_1['assignment_id'], assignment_id_1)
        self.assertEqual(pair_2['assignment_id'], assignment_id_2)
        self.assertEqual(pair_3['assignment_id'], assignment_id_3)
        self.assertEqual(pair_1['run_id'], run_id)
        self.assertEqual(pair_2['run_id'], run_id)
        self.assertEqual(pair_3['run_id'], run_id)
        self.assertEqual(pair_1['bonus_amount'], 0)
        self.assertEqual(pair_2['bonus_amount'], 0)
        self.assertEqual(pair_3['bonus_amount'], 0)
        self.assertEqual(pair_1['bonus_text'], '')
        self.assertEqual(pair_2['bonus_text'], '')
        self.assertEqual(pair_3['bonus_text'], '')
        self.assertFalse(pair_1['bonus_paid'])
        self.assertFalse(pair_2['bonus_paid'])
        self.assertFalse(pair_3['bonus_paid'])

        # Ensure get_pairings_for_assignment works
        pair_4 = db_logger.get_pairings_for_assignment(assignment_id_2)[0]
        self.assertEqual(pair_2, pair_4)
        self.assertListEqual(
            [], db_logger.get_pairings_for_assignment('fake_id'))

        # Ensure get_all_<thing>_for_worker works
        self.assertListEqual(
            [], db_logger.get_all_assignments_for_worker('fake_id'))
        self.assertListEqual(
            [], db_logger.get_all_pairings_for_worker('fake_id'))
        self.assertEqual(
            db_logger.get_all_assignments_for_worker(worker_id_1)[0],
            assignment_1_data)
        self.assertEqual(
            len(db_logger.get_all_assignments_for_worker(worker_id_2)), 2)
        self.assertEqual(
            db_logger.get_all_pairings_for_worker(worker_id_1)[0],
            pair_1)
        self.assertEqual(
            len(db_logger.get_all_pairings_for_worker(worker_id_2)), 2)

        # test task_restricted gets
        self.assertEqual(
            db_logger.get_all_task_assignments_for_worker(worker_id_1)[0],
            assignment_1_data)
        self.assertEqual(
            len(db_logger.get_all_task_assignments_for_worker(worker_id_2)), 2)
        self.assertEqual(
            len(db_logger.get_all_task_assignments_for_worker(
                worker_id_1, 'fake_id')), 0)
        self.assertEqual(
            db_logger.get_all_task_pairings_for_worker(worker_id_1)[0],
            pair_1)
        self.assertEqual(
            len(db_logger.get_all_task_pairings_for_worker(worker_id_2)), 2)
        self.assertEqual(
            len(db_logger.get_all_task_pairings_for_worker(
                worker_id_1, 'fake_id')), 0)

        conversation_id_1 = "CONV_ID_1"
        conversation_id_2 = "CONV_ID_2"

        onboarding_id_1 = 'onboard_1'
        onboarding_id_2 = 'onboard_2'
        onboarding_id_3 = 'onboard_3'

        db_logger.log_start_onboard(
            worker_id_1, assignment_id_1, onboarding_id_1)
        db_logger.log_start_onboard(
            worker_id_2, assignment_id_2, onboarding_id_2)
        db_logger.log_start_onboard(
            worker_id_2, assignment_id_3, onboarding_id_3)
        db_logger.log_finish_onboard(worker_id_1, assignment_id_1)
        db_logger.log_finish_onboard(worker_id_2, assignment_id_2)
        db_logger.log_finish_onboard(worker_id_2, assignment_id_3)
        db_logger.log_start_task(worker_id_1, assignment_id_1,
                                 conversation_id_1)
        db_logger.log_start_task(worker_id_2, assignment_id_2,
                                 conversation_id_1)
        db_logger.log_start_task(worker_id_2, assignment_id_3,
                                 conversation_id_2)

        # Check to see retrieval by conversation
        pairs_1 = db_logger.get_pairings_for_conversation(conversation_id_1)
        pairs_2 = db_logger.get_pairings_for_conversation(conversation_id_2)
        pairs_3 = db_logger.get_pairings_for_conversation('fake_id')
        pair_1 = db_logger.get_worker_assignment_pairing(
            worker_id_1, assignment_id_1)
        pair_2 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_2)
        pair_3 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_3)
        self.assertEqual(pairs_1[0], pair_1)
        self.assertEqual(pairs_1[1], pair_2)
        self.assertEqual(pairs_2[0], pair_3)
        self.assertEqual(len(pairs_3), 0)

        # Do some final processing on assignments
        db_logger.log_complete_assignment(
            worker_id_1, assignment_id_1, time.time(),
            AssignState.STATUS_PARTNER_DISCONNECT)
        db_logger.log_disconnect_assignment(
            worker_id_2, assignment_id_2, time.time(),
            AssignState.STATUS_DISCONNECT)
        db_logger.log_complete_assignment(worker_id_2, assignment_id_3,
                                          time.time(), AssignState.STATUS_DONE)

        # Assignment state consistent
        assignment_1_data = db_logger.get_assignment_data(assignment_id_1)
        assignment_2_data = db_logger.get_assignment_data(assignment_id_2)
        assignment_3_data = db_logger.get_assignment_data(assignment_id_3)
        self.assertEqual(assignment_1_data['assignment_id'], assignment_id_1)
        self.assertEqual(assignment_1_data['status'], 'Completed')
        self.assertIsNotNone(assignment_1_data['approve_time'])
        self.assertEqual(assignment_1_data['worker_id'], worker_id_1)
        self.assertEqual(assignment_1_data['hit_id'], HIT1['HIT']['HITId'])
        self.assertEqual(assignment_2_data['assignment_id'], assignment_id_2)
        self.assertEqual(assignment_2_data['status'], 'Disconnected')
        self.assertIsNotNone(assignment_2_data['approve_time'])
        self.assertEqual(assignment_2_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_2_data['hit_id'], HIT2['HIT']['HITId'])
        self.assertEqual(assignment_3_data['assignment_id'], assignment_id_3)
        self.assertEqual(assignment_3_data['status'], 'Completed')
        self.assertIsNotNone(assignment_3_data['approve_time'])
        self.assertEqual(assignment_3_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_3_data['hit_id'], HIT3['HIT']['HITId'])

        # Worker state consistent
        worker_1_data = db_logger.get_worker_data(worker_id_1)
        worker_2_data = db_logger.get_worker_data(worker_id_2)
        self.assertEqual(worker_1_data['worker_id'], worker_id_1)
        self.assertEqual(worker_1_data['accepted'], 1)
        self.assertEqual(worker_1_data['disconnected'], 0)
        self.assertEqual(worker_1_data['completed'], 1)
        self.assertEqual(worker_1_data['approved'], 0)
        self.assertEqual(worker_1_data['rejected'], 0)
        self.assertEqual(worker_2_data['worker_id'], worker_id_2)
        self.assertEqual(worker_2_data['accepted'], 2)
        self.assertEqual(worker_2_data['disconnected'], 1)
        self.assertEqual(worker_2_data['completed'], 1)
        self.assertEqual(worker_2_data['approved'], 0)
        self.assertEqual(worker_2_data['rejected'], 0)

        # Ensure Pairing state is consistent
        pair_1 = db_logger.get_worker_assignment_pairing(
            worker_id_1, assignment_id_1)
        pair_2 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_2)
        pair_3 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_3)
        self.assertEqual(pair_1['status'],
                         AssignState.STATUS_PARTNER_DISCONNECT)
        self.assertEqual(pair_2['status'], AssignState.STATUS_DISCONNECT)
        self.assertEqual(pair_3['status'], AssignState.STATUS_DONE)
        self.assertEqual(pair_1['worker_id'], worker_id_1)
        self.assertEqual(pair_2['worker_id'], worker_id_2)
        self.assertEqual(pair_3['worker_id'], worker_id_2)
        self.assertEqual(pair_1['assignment_id'], assignment_id_1)
        self.assertEqual(pair_2['assignment_id'], assignment_id_2)
        self.assertEqual(pair_3['assignment_id'], assignment_id_3)
        self.assertEqual(pair_1['conversation_id'], conversation_id_1)
        self.assertEqual(pair_2['conversation_id'], conversation_id_1)
        self.assertEqual(pair_3['conversation_id'], conversation_id_2)
        self.assertGreaterEqual(pair_1['onboarding_end'],
                                pair_1['onboarding_start'])
        self.assertGreaterEqual(pair_2['onboarding_end'],
                                pair_2['onboarding_start'])
        self.assertGreaterEqual(pair_3['onboarding_end'],
                                pair_3['onboarding_start'])
        self.assertGreaterEqual(pair_1['task_start'],
                                pair_1['onboarding_end'])
        self.assertGreaterEqual(pair_2['task_start'],
                                pair_2['onboarding_end'])
        self.assertGreaterEqual(pair_3['task_start'],
                                pair_3['onboarding_end'])
        self.assertGreaterEqual(pair_1['task_end'],
                                pair_1['onboarding_start'])
        self.assertGreaterEqual(pair_2['task_end'],
                                pair_2['onboarding_start'])
        self.assertGreaterEqual(pair_3['task_end'],
                                pair_3['onboarding_start'])
        self.assertEqual(pair_1['run_id'], run_id)
        self.assertEqual(pair_2['run_id'], run_id)
        self.assertEqual(pair_3['run_id'], run_id)
        self.assertEqual(pair_1['bonus_amount'], 0)
        self.assertEqual(pair_2['bonus_amount'], 0)
        self.assertEqual(pair_3['bonus_amount'], 0)
        self.assertEqual(pair_1['bonus_text'], '')
        self.assertEqual(pair_2['bonus_text'], '')
        self.assertEqual(pair_3['bonus_text'], '')
        self.assertFalse(pair_1['bonus_paid'])
        self.assertFalse(pair_2['bonus_paid'])
        self.assertFalse(pair_3['bonus_paid'])

        # Ensure run state is consistent
        run_data = db_logger.get_run_data(run_id)
        self.assertEqual(run_data['run_id'], run_id)
        self.assertEqual(run_data['created'], 3)
        self.assertEqual(run_data['completed'], 2)
        self.assertEqual(run_data['maximum'], hits_created)
        self.assertEqual(run_data['failed'], 1)

        # Test "submitting" and abandoning hits
        db_logger.log_submit_assignment(worker_id_1, assignment_id_1)
        db_logger.log_abandon_assignment(worker_id_2, assignment_id_2)
        db_logger.log_expire_assignment(worker_id_2, assignment_id_3)
        assignment_1_data = db_logger.get_assignment_data(assignment_id_1)
        assignment_2_data = db_logger.get_assignment_data(assignment_id_2)
        assignment_3_data = db_logger.get_assignment_data(assignment_id_3)
        self.assertEqual(assignment_1_data['assignment_id'], assignment_id_1)
        self.assertEqual(assignment_1_data['status'], 'Reviewable')
        self.assertIsNotNone(assignment_1_data['approve_time'])
        self.assertEqual(assignment_1_data['worker_id'], worker_id_1)
        self.assertEqual(assignment_1_data['hit_id'], HIT1['HIT']['HITId'])
        self.assertEqual(assignment_2_data['assignment_id'], assignment_id_2)
        self.assertEqual(assignment_2_data['status'], 'Abandoned')
        self.assertIsNotNone(assignment_2_data['approve_time'])
        self.assertEqual(assignment_2_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_2_data['hit_id'], HIT2['HIT']['HITId'])
        self.assertEqual(assignment_3_data['assignment_id'], assignment_id_3)
        self.assertEqual(assignment_3_data['status'], 'Expired')
        self.assertIsNotNone(assignment_3_data['approve_time'])
        self.assertEqual(assignment_3_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_3_data['hit_id'], HIT3['HIT']['HITId'])

        # Test approving and rejecting
        test_dollars = 3
        test_cents = 100 * test_dollars
        reason_use = 'Just because'
        out_reason = '${} for {}\n'.format(test_dollars, reason_use)
        db_logger.log_award_amount(worker_id_1, assignment_id_1, test_dollars,
                                   reason_use)
        db_logger.log_award_amount(worker_id_2, assignment_id_2, test_dollars,
                                   reason_use)
        db_logger.log_bonus_paid(worker_id_1, assignment_id_1)
        db_logger.log_approve_assignment(assignment_id_1)
        db_logger.log_reject_assignment(assignment_id_2)

        # Ensure state is valid again
        assignment_1_data = db_logger.get_assignment_data(assignment_id_1)
        assignment_2_data = db_logger.get_assignment_data(assignment_id_2)
        self.assertEqual(assignment_1_data['assignment_id'], assignment_id_1)
        self.assertEqual(assignment_1_data['status'], 'Approved')
        self.assertIsNotNone(assignment_1_data['approve_time'])
        self.assertEqual(assignment_1_data['worker_id'], worker_id_1)
        self.assertEqual(assignment_1_data['hit_id'], HIT1['HIT']['HITId'])
        self.assertEqual(assignment_2_data['assignment_id'], assignment_id_2)
        self.assertEqual(assignment_2_data['status'], 'Rejected')
        self.assertIsNotNone(assignment_2_data['approve_time'])
        self.assertEqual(assignment_2_data['worker_id'], worker_id_2)
        self.assertEqual(assignment_2_data['hit_id'], HIT2['HIT']['HITId'])

        worker_1_data = db_logger.get_worker_data(worker_id_1)
        worker_2_data = db_logger.get_worker_data(worker_id_2)
        self.assertEqual(worker_1_data['worker_id'], worker_id_1)
        self.assertEqual(worker_1_data['accepted'], 1)
        self.assertEqual(worker_1_data['disconnected'], 0)
        self.assertEqual(worker_1_data['expired'], 0)
        self.assertEqual(worker_1_data['completed'], 1)
        self.assertEqual(worker_1_data['approved'], 1)
        self.assertEqual(worker_1_data['rejected'], 0)
        self.assertEqual(worker_2_data['worker_id'], worker_id_2)
        self.assertEqual(worker_2_data['accepted'], 2)
        self.assertEqual(worker_2_data['disconnected'], 2)
        self.assertEqual(worker_2_data['expired'], 1)
        self.assertEqual(worker_2_data['completed'], 1)
        self.assertEqual(worker_2_data['approved'], 0)
        self.assertEqual(worker_2_data['rejected'], 1)

        pair_1 = db_logger.get_worker_assignment_pairing(
            worker_id_1, assignment_id_1)
        pair_2 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_2)
        pair_3 = db_logger.get_worker_assignment_pairing(
            worker_id_2, assignment_id_3)
        self.assertEqual(pair_1['status'],
                         AssignState.STATUS_PARTNER_DISCONNECT)
        self.assertEqual(pair_2['status'], AssignState.STATUS_DISCONNECT)
        self.assertEqual(pair_3['status'], AssignState.STATUS_EXPIRED)
        self.assertEqual(pair_1['worker_id'], worker_id_1)
        self.assertEqual(pair_2['worker_id'], worker_id_2)
        self.assertEqual(pair_3['worker_id'], worker_id_2)
        self.assertEqual(pair_1['assignment_id'], assignment_id_1)
        self.assertEqual(pair_2['assignment_id'], assignment_id_2)
        self.assertEqual(pair_3['assignment_id'], assignment_id_3)
        self.assertEqual(pair_1['conversation_id'], conversation_id_1)
        self.assertEqual(pair_2['conversation_id'], conversation_id_1)
        self.assertEqual(pair_3['conversation_id'], conversation_id_2)
        self.assertGreaterEqual(pair_1['onboarding_end'],
                                pair_1['onboarding_start'])
        self.assertGreaterEqual(pair_2['onboarding_end'],
                                pair_2['onboarding_start'])
        self.assertGreaterEqual(pair_3['onboarding_end'],
                                pair_3['onboarding_start'])
        self.assertGreaterEqual(pair_1['task_start'],
                                pair_1['onboarding_end'])
        self.assertGreaterEqual(pair_2['task_start'],
                                pair_2['onboarding_end'])
        self.assertGreaterEqual(pair_3['task_start'],
                                pair_3['onboarding_end'])
        self.assertGreaterEqual(pair_1['task_end'],
                                pair_1['onboarding_start'])
        self.assertGreaterEqual(pair_2['task_end'],
                                pair_2['onboarding_start'])
        self.assertGreaterEqual(pair_3['task_end'],
                                pair_3['onboarding_start'])
        self.assertEqual(pair_1['run_id'], run_id)
        self.assertEqual(pair_2['run_id'], run_id)
        self.assertEqual(pair_3['run_id'], run_id)
        self.assertEqual(pair_1['bonus_amount'], test_cents)
        self.assertEqual(pair_2['bonus_amount'], test_cents)
        self.assertEqual(pair_3['bonus_amount'], 0)
        self.assertEqual(pair_1['bonus_text'], out_reason)
        self.assertEqual(pair_2['bonus_text'], out_reason)
        self.assertEqual(pair_3['bonus_text'], '')
        self.assertTrue(pair_1['bonus_paid'])
        self.assertFalse(pair_2['bonus_paid'])
        self.assertFalse(pair_3['bonus_paid'])


if __name__ == '__main__':
    unittest.main()
