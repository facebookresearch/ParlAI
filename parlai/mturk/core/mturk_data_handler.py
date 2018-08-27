# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import os
import sqlite3
import time

import parlai.mturk.core.shared_utils as shared_utils
from parlai.mturk.core.agents import AssignState

parent_dir = os.path.dirname(os.path.abspath(__file__))

# Run data table:
CREATE_RUN_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS runs (
        id string PRIMARY KEY,
        created integer NOT NULL,
        maximum integer NOT NULL,
        completed integer NOT NULL,
        failed integer NOT NULL,
    );
    """)

# Worker data table:
CREATE_WORKER_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS workers (
        id string PRIMARY KEY,
        accepted integer NOT NULL,
        disconnected integer NOT NULL,
        completed integer NOT NULL,
        approved integer NOT NULL,
        rejected integer NOT NULL
    );
    """)

# HIT data table:
CREATE_HIT_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS hits (
        id string PRIMARY KEY,
        expiration integer NOT NULL,
        hit_status string,
        assignments_pending int,
        assignments_available int,
        assignments_complete int,
        FOREIGN KEY (run_id) REFERENCES runs (id)
    );
    """)

# Assignment data table: (as one HIT can technically have multiple assignments)
CREATE_ASSIGN_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS assignments (
        id string PRIMARY KEY,
        status string,
        approve_time int,
        FOREIGN KEY (worker_id) REFERENCES workers (id),
        FOREIGN KEY (hit_id) REFERENCES hits (id),
    );
    """)

# pairing data table: (reflects one worker<->assignment pairing)
CREATE_PAIRING_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS pairings (
        status string,
        onboarding_start int,
        onboarding_end int,
        task_start int,
        task_end int,
        conversation_id string,
        bonus_amount int,
        bonus_text string,
        bonus_paid boolean,
        FOREIGN KEY (worker_id) REFERENCES workers (id),
        FOREIGN KEY (assignment_id) REFERENCES assignments (id),
    );
    """)


class MTurkDataHandler():
    """Handles logging data to and reading data from a SQLite3 table for
    observation across processes and for controlled restarts
    """
    def __init__(self, task_group_id=None, file_name='pmt_data.db'):
        self.db_path = os.path.join(parent_dir, file_name)
        self.conn = None
        self.task_group_id = task_group_id
        self.create_default_tables()

    def _get_connection(self):
        if self.conn is None:
            try:
                conn = sqlite3.connect(self.db_path)
                self.conn = conn
            except sqlite3.Error as e:
                shared_utils.print_and_log(
                    logging.ERROR,
                    "Could not get db connection, failing: {}".format(repr(e)),
                    should_print=True)
                raise e
        return self.conn

    def create_default_tables(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute(CREATE_RUN_DATA_SQL_TABLE)
        c.execute(CREATE_WORKER_DATA_SQL_TABLE)
        c.execute(CREATE_HIT_DATA_SQL_TABLE)
        c.execute(CREATE_ASSIGN_DATA_SQL_TABLE)
        c.execute(CREATE_PAIRING_DATA_SQL_TABLE)
        conn.commit()

    def log_new_run(self, target_hits, task_group_id=None):
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO runs VALUES (?,?,?,?,?);',
                  (task_group_id, 0, target_hits, 0, 0))
        conn.commit()

    def log_hit_status(self, mturk_hit_creation_response, task_group_id=None):
        '''Create or update an entry in the hit sattus table'''
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        hit_details = mturk_hit_creation_response['HIT']
        id = hit_details['HITId']
        expiration = time.mktime(hit_details['Expiration'].timetuple())
        status = hit_details['HITStatus']
        assignments_pending = hit_details['NumberOfAssignmentsPending']
        assignments_available = hit_details['NumberOfAssignmentsAvailable']
        assignments_complete = hit_details['NumberOfAssignmentsCompleted']
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('REPLACE INTO hits VALUES (?,?,?,?,?,?,?);',
                  (id, expiration, status, assignments_pending,
                   assignments_available, assignments_complete, task_group_id))
        c.execute('UPDATE runs SET created = created + 1 WHERE id = ?;',
                  (task_group_id, ))
        conn.commit()

    def log_worker_accept_assignment(self, worker_id, assignment_id, hit_id):
        conn = self._get_connection()
        c = conn.cursor()

        # Ensure worker exists, mark the accepted assignment
        c.execute('SELECT * FROM workers WHERE id = ?;', (worker_id, ))
        has_worker = c.rowcount > 0
        if not has_worker:
            # Must instert a new worker into the database
            c.execute('INSERT INTO workers VALUES (?,?,?,?,?,?);',
                      (worker_id, 1, 0, 0, 0, 0))
        else:
            # Increment number of assignments the worker has accepted
            c.execute(
                'UPDATE workers SET accepted = accepted + 1 WHERE id = ?;',
                (worker_id, )
            )

        # Ensure the assignment exsits, mark the current worker
        c.execute('REPLACE INTO assignments VALUES (?,?,?,?,?)',
                  (assignment_id, 'Accepted', None, worker_id, hit_id))

        # Create tracking for this specific pairing, as the assignment may be
        # reassigned
        c.execute('INSERT INTO pairings VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                  (AssignState.STATUS_NONE, None, None, None, None, None, 0,
                   '', False, worker_id, assignment_id))
        conn.commit()

    def log_complete_assignment(self, worker_id, assignment_id, approve_time,
                                complete_type, task_group_id=None):
        '''Note that an assignment was completed'''
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        conn = self._get_connection()
        c = conn.cursor()
        # Update assign data to reviewable
        c.execute('UPDATE assignments SET status = ? WHERE assignment_id = ?;',
                  ('Reviewable', assignment_id))

        # Increment worker completed
        c.execute('UPDATE workers SET completed = completed + 1 WHERE id = ?;',
                  (worker_id, ))

        # update the payment data status
        c.execute('''UPDATE pairings SET status = ?, task_end = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (complete_type, time.time(), worker_id, assignment_id))

        # Update run data to have another completed
        c.execute('UPDATE runs SET completed = completed + 1 WHERE id = ?;',
                  (task_group_id, 1))
        conn.commit()

    def log_abandoned_assignment(self, worker_id, assignment_id, approve_time,
                                 disconnect_type, task_group_id=None):
        '''Note that an assignment was completed'''
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        conn = self._get_connection()
        c = conn.cursor()

        # Update assign data to reviewable
        c.execute('UPDATE assignments SET status = ? WHERE assignment_id = ?;',
                  ('Reviewable', assignment_id))

        # Increment worker completed
        c.execute('''UPDATE workers SET disconnected = disconnected + 1
                     WHERE id = ?;''',
                  (worker_id, ))

        # update the payment data status
        c.execute('''UPDATE pairings SET status = ?, task_end = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (disconnect_type, time.time(), worker_id, assignment_id))

        # Update run data to have another completed
        c.execute('UPDATE runs SET failed = failed + 1 WHERE id = ?;',
                  (task_group_id, 1))
        conn.commit()

    def log_start_onboard(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET status = ?, onboarding_start = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (AssignState.STATUS_ONBOARDING, time.time(), worker_id,
                   assignment_id))
        conn.commit()

    def log_finish_onboard(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET status = ?, onboarding_end = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (AssignState.STATUS_WAITING, time.time(), worker_id,
                   assignment_id))
        conn.commit()

    def log_start_task(self, worker_id, assignment_id, conversation_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET status = ?, task_start = ?,
                     conversation_id = ? WHERE worker_id = ?
                     AND assignment_id = ?;''',
                  (AssignState.STATUS_IN_TASK, time.time(), conversation_id,
                   worker_id, assignment_id))
        conn.commit()

    def log_award_amount(self, worker_id, assignment_id, amount, reason):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET bonus_amount = ?, bonus_text = ?,
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (amount, reason, worker_id, assignment_id))
        conn.commit()

    def log_bonus_paid(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET bonus_paid = ?,
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (True, worker_id, assignment_id))
        conn.commit()

    def log_approve_assignment(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ? WHERE assignment_id = ?;',
                  ('Approved', assignment_id))
        c.execute('''UPDATE workers SET approved = approved + 1
                     WHERE worker_id = ?;''',
                  (worker_id, ))
        conn.commit()

    def log_reject_assignment(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ? WHERE assignment_id = ?;',
                  ('Rejected', assignment_id))
        c.execute('''UPDATE workers SET rejected = rejected + 1
                     WHERE worker_id = ?;''',
                  (worker_id, ))
        conn.commit()
