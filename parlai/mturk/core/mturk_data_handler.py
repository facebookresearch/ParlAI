#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import os
import sqlite3
import time
import threading

import parlai.mturk.core.shared_utils as shared_utils
from parlai.mturk.core.agents import AssignState

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/run_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Run data table:
CREATE_RUN_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS runs (
        run_id string PRIMARY KEY,
        created integer NOT NULL,
        maximum integer NOT NULL,
        completed integer NOT NULL,
        failed integer NOT NULL
    );
    """)

# Worker data table:
CREATE_WORKER_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS workers (
        worker_id string PRIMARY KEY,
        accepted integer NOT NULL,
        disconnected integer NOT NULL,
        expired integer NOT NULL,
        completed integer NOT NULL,
        approved integer NOT NULL,
        rejected integer NOT NULL
    );
    """)

# HIT data table:
CREATE_HIT_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS hits (
        hit_id string PRIMARY KEY,
        expiration integer NOT NULL,
        hit_status string,
        assignments_pending int,
        assignments_available int,
        assignments_complete int,
        run_id string,
        FOREIGN KEY (run_id) REFERENCES runs (run_id)
    );
    """)

# Assignment data table: (as one HIT can technically have multiple assignments)
CREATE_ASSIGN_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS assignments (
        assignment_id string PRIMARY KEY,
        status string,
        approve_time int,
        worker_id string,
        hit_id string,
        FOREIGN KEY (worker_id) REFERENCES workers (worker_id),
        FOREIGN KEY (hit_id) REFERENCES hits (hit_id)
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
        notes string,
        worker_id string,
        assignment_id string,
        run_id string,
        FOREIGN KEY (worker_id) REFERENCES workers (worker_id),
        FOREIGN KEY (assignment_id) REFERENCES assignments (assignment_id),
        FOREIGN KEY (run_id) REFERENCES runs (run_id)
    );
    """)


class MTurkDataHandler():
    """Handles logging data to and reading data from a SQLite3 table for
    observation across processes and for controlled restarts
    """
    def __init__(self, task_group_id=None, file_name='pmt_data.db'):
        self.db_path = os.path.join(data_dir, file_name)
        self.conn = {}
        self.task_group_id = task_group_id
        self.table_access_condition = threading.Condition()
        self.create_default_tables()

    def _get_connection(self):
        '''Returns a singular database connection to be shared amongst all
        calls
        '''
        curr_thread = threading.get_ident()
        if curr_thread not in self.conn or self.conn[curr_thread] is None:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                self.conn[curr_thread] = conn
            except sqlite3.Error as e:
                shared_utils.print_and_log(
                    logging.ERROR,
                    "Could not get db connection, failing: {}".format(repr(e)),
                    should_print=True)
                raise e
        return self.conn[curr_thread]

    def _force_task_group_id(self, task_group_id):
        '''Throw an error if a task group id is neither provided nor stored'''
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'
        return task_group_id

    def create_default_tables(self):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute(CREATE_RUN_DATA_SQL_TABLE)
            c.execute(CREATE_WORKER_DATA_SQL_TABLE)
            c.execute(CREATE_HIT_DATA_SQL_TABLE)
            c.execute(CREATE_ASSIGN_DATA_SQL_TABLE)
            c.execute(CREATE_PAIRING_DATA_SQL_TABLE)
            conn.commit()

    def log_new_run(self, target_hits, task_group_id=None):
        with self.table_access_condition:
            task_group_id = self._force_task_group_id(task_group_id)
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('INSERT INTO runs VALUES (?,?,?,?,?);',
                      (task_group_id, 0, target_hits, 0, 0))
            conn.commit()

    def log_hit_status(self, mturk_hit_creation_response, task_group_id=None):
        '''Create or update an entry in the hit status table'''
        task_group_id = self._force_task_group_id(task_group_id)

        hit_details = mturk_hit_creation_response['HIT']
        id = hit_details['HITId']
        expiration = time.mktime(hit_details['Expiration'].timetuple())
        status = hit_details['HITStatus']
        assignments_pending = hit_details['NumberOfAssignmentsPending']
        assignments_available = hit_details['NumberOfAssignmentsAvailable']
        assignments_complete = hit_details['NumberOfAssignmentsCompleted']
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM hits WHERE hit_id = ?;', (id, ))
            is_new_hit = c.fetchone()[0] == 0
            if is_new_hit:
                c.execute('''UPDATE runs SET created = created + 1
                             WHERE run_id = ?;''',
                          (task_group_id, ))

            c.execute('REPLACE INTO hits VALUES (?,?,?,?,?,?,?);',
                      (id, expiration, status, assignments_pending,
                       assignments_available, assignments_complete,
                       task_group_id))
            conn.commit()

    def log_worker_accept_assignment(self, worker_id, assignment_id, hit_id,
                                     task_group_id=None):
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()

            # Ensure worker exists, mark the accepted assignment
            c.execute('SELECT COUNT(*) FROM workers WHERE worker_id = ?;',
                      (worker_id, ))
            has_worker = c.fetchone()[0] > 0
            if not has_worker:
                # Must instert a new worker into the database
                c.execute('INSERT INTO workers VALUES (?,?,?,?,?,?,?);',
                          (worker_id, 1, 0, 0, 0, 0, 0))
            else:
                # Increment number of assignments the worker has accepted
                c.execute('''UPDATE workers SET accepted = accepted + 1
                             WHERE worker_id = ?;''',
                          (worker_id, ))

            # Ensure the assignment exists, mark the current worker
            c.execute('REPLACE INTO assignments VALUES (?,?,?,?,?)',
                      (assignment_id, 'Accepted', None, worker_id, hit_id))

            # Create tracking for this specific pairing, as the assignment
            # may be reassigned
            c.execute('''INSERT INTO pairings
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                      (AssignState.STATUS_NONE, None, None, None, None, None,
                       0, '', False, '', worker_id, assignment_id,
                       task_group_id))
            conn.commit()

    def log_complete_assignment(self, worker_id, assignment_id, approve_time,
                                complete_type, task_group_id=None):
        '''Note that an assignment was completed'''
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            # Update assign data to completed
            c.execute('''UPDATE assignments SET status = ?, approve_time = ?
                         WHERE assignment_id = ?;''',
                      ('Completed', approve_time, assignment_id))

            # Increment worker completed
            c.execute('''UPDATE workers SET completed = completed + 1
                         WHERE worker_id = ?;''',
                      (worker_id, ))

            # update the payment data status
            c.execute('''UPDATE pairings SET status = ?, task_end = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (complete_type, time.time(), worker_id, assignment_id))

            # Update run data to have another completed
            c.execute('''UPDATE runs SET completed = completed + 1
                         WHERE run_id = ?;''',
                      (task_group_id, ))
            conn.commit()

    def log_disconnect_assignment(self, worker_id, assignment_id, approve_time,
                                  disconnect_type, task_group_id=None):
        '''Note that an assignment was disconnected from'''
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()

            # Update assign data to completed for this task (we can't track)
            c.execute('''UPDATE assignments SET status = ?, approve_time = ?
                         WHERE assignment_id = ?;''',
                      ('Completed', approve_time, assignment_id))

            # Increment worker completed
            c.execute('''UPDATE workers SET disconnected = disconnected + 1
                         WHERE worker_id = ?;''',
                      (worker_id, ))

            # update the pairing status
            c.execute('''UPDATE pairings SET status = ?, task_end = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (disconnect_type, time.time(), worker_id, assignment_id))

            # Update run data to have another completed
            c.execute('UPDATE runs SET failed = failed + 1 WHERE run_id = ?;',
                      (task_group_id, ))
            conn.commit()

    def log_expire_assignment(self, worker_id, assignment_id,
                              task_group_id=None):
        '''Note that an assignment was expired by us'''
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()

            # Update assign data to expired
            c.execute('''UPDATE assignments SET status = ?
                         WHERE assignment_id = ?;''',
                      ('Expired', assignment_id))

            # Increment worker completed
            c.execute('''UPDATE workers SET expired = expired + 1
                         WHERE worker_id = ?;''',
                      (worker_id, ))

            # update the pairing status
            c.execute('''UPDATE pairings SET status = ?, task_end = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (AssignState.STATUS_EXPIRED, time.time(), worker_id,
                       assignment_id))

            # Update run data to have another completed
            c.execute('UPDATE runs SET failed = failed + 1 WHERE run_id = ?;',
                      (task_group_id, ))
            conn.commit()

    def log_submit_assignment(self, worker_id, assignment_id):
        '''To be called whenever a worker hits the "submit hit" button'''
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            # update the assignment status to reviewable
            c.execute('''UPDATE assignments SET status = ?
                         WHERE assignment_id = ?;''',
                      ('Reviewable', assignment_id))
            conn.commit()

    def log_abandon_assignment(self, worker_id, assignment_id):
        '''To be called whenever a worker returns a hit'''
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            # update the assignment status to reviewable
            c.execute('''UPDATE assignments SET status = ?
                         WHERE assignment_id = ?;''',
                      ('Abandoned', assignment_id))
            conn.commit()

    def log_start_onboard(self, worker_id, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE pairings SET status = ?, onboarding_start = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (AssignState.STATUS_ONBOARDING, time.time(), worker_id,
                       assignment_id))
            conn.commit()

    def log_finish_onboard(self, worker_id, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE pairings SET status = ?, onboarding_end = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (AssignState.STATUS_WAITING, time.time(), worker_id,
                       assignment_id))
            conn.commit()

    def log_start_task(self, worker_id, assignment_id, conversation_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE pairings SET status = ?, task_start = ?,
                         conversation_id = ? WHERE worker_id = ?
                         AND assignment_id = ?;''',
                      (AssignState.STATUS_IN_TASK, time.time(),
                       conversation_id, worker_id, assignment_id))
            conn.commit()

    def log_award_amount(self, worker_id, assignment_id, amount, reason):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE pairings SET bonus_amount = ?, bonus_text = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (amount, reason, worker_id, assignment_id))
            conn.commit()

    def log_bonus_paid(self, worker_id, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE pairings SET bonus_paid = ?
                         WHERE worker_id = ? AND assignment_id = ?;''',
                      (True, worker_id, assignment_id))
            conn.commit()

    def log_approve_assignment(self, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE assignments SET status = ?
                         WHERE assignment_id = ?;''',
                      ('Approved', assignment_id))
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;',
                      (assignment_id, ))
            assignment = c.fetchone()
            if assignment is None:
                return
            worker_id = assignment['worker_id']
            c.execute('''UPDATE workers SET approved = approved + 1
                         WHERE worker_id = ?;''',
                      (worker_id, ))
            conn.commit()

    def log_reject_assignment(self, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''UPDATE assignments SET status = ?
                         WHERE assignment_id = ?;''',
                      ('Rejected', assignment_id))
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;',
                      (assignment_id, ))
            assignment = c.fetchone()
            if assignment is None:
                return
            worker_id = assignment['worker_id']
            c.execute('''UPDATE workers SET rejected = rejected + 1
                         WHERE worker_id = ?;''',
                      (worker_id, ))
            conn.commit()

    def log_worker_note(self, worker_id, assignment_id, note):
        note += '\n'
        with self.table_access_condition:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                c.execute('''UPDATE pairings SET notes = notes || ?
                             WHERE worker_id = ? AND assignment_id = ?;''',
                          (note, worker_id, assignment_id))
                conn.commit()
            except Exception as e:
                print(repr(e))

    def get_worker_data(self, worker_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM workers WHERE worker_id = ?;',
                      (worker_id, ))
            results = c.fetchone()
            return results

    def get_assignment_data(self, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;',
                      (assignment_id, ))
            results = c.fetchone()
            return results

    def get_worker_assignment_pairing(self, worker_id, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("""SELECT * FROM pairings WHERE worker_id = ?
                         AND assignment_id = ?;""",
                      (worker_id, assignment_id))
            results = c.fetchone()
            return results

    def get_run_data(self, task_group_id):
        '''get the run data for the given task_group_id, return None if not
        found.
        '''
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM runs WHERE run_id = ?;',
                      (task_group_id, ))
            results = c.fetchone()
            return results

    def get_hit_data(self, hit_id):
        '''get the hit data for the given hit_id, return None if not'''
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM hits WHERE hit_id = ?;", (hit_id, ))
            results = c.fetchone()
            return results

    def get_pairings_for_assignment(self, assignment_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM pairings WHERE assignment_id = ?;",
                      (assignment_id, ))
            results = c.fetchall()
            return results

    def get_pairings_for_conversation(self, conversation_id,
                                      task_group_id=None):
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("""SELECT * FROM pairings WHERE conversation_id = ?
                         AND run_id = ?;""", (conversation_id, task_group_id))
            results = c.fetchall()
            return results

    def get_all_assignments_for_worker(self, worker_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM assignments WHERE worker_id = ?;",
                      (worker_id, ))
            results = c.fetchall()
            return results

    def get_all_pairings_for_worker(self, worker_id):
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE worker_id = ?;',
                      (worker_id, ))
            results = c.fetchall()
            return results

    def get_all_task_assignments_for_worker(self, worker_id,
                                            task_group_id=None):
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("""SELECT assignments.assignment_id, assignments.status,
                         assignments.approve_time, assignments.worker_id,
                         assignments.hit_id
                         FROM assignments
                         INNER JOIN hits on assignments.hit_id = hits.hit_id
                         WHERE assignments.worker_id = ? AND hits.run_id = ?;
                         """,
                      (worker_id, task_group_id))
            results = c.fetchall()
            return results

    def get_all_task_pairings_for_worker(self, worker_id, task_group_id=None):
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''SELECT * FROM pairings WHERE worker_id = ?
                         AND run_id = ?;''',
                      (worker_id, task_group_id))
            results = c.fetchall()
            return results
