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
        failed integer NOT NULL
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
        run_id string,
        FOREIGN KEY (run_id) REFERENCES runs (id)
    );
    """)

# Assignment data table: (as one HIT can technically have multiple assignments)
CREATE_ASSIGN_DATA_SQL_TABLE = (
    """CREATE TABLE IF NOT EXISTS assignments (
        id string PRIMARY KEY,
        status string,
        approve_time int,
        worker_id string,
        hit_id string,
        FOREIGN KEY (worker_id) REFERENCES workers (id),
        FOREIGN KEY (hit_id) REFERENCES hits (id)
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
        worker_id string,
        assignment_id string,
        run_id string,
        FOREIGN KEY (worker_id) REFERENCES workers (id),
        FOREIGN KEY (assignment_id) REFERENCES assignments (id),
        FOREIGN KEY (run_id) REFERENCES runs (id)
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

    def _format_pairing(self, pairing_result):
        return {
            'status': pairing_result[0],
            'onboarding_start': pairing_result[1],
            'onboarding_end': pairing_result[2],
            'task_start': pairing_result[3],
            'task_end': pairing_result[4],
            'conversation_id': pairing_result[5],
            'bonus_amount': pairing_result[6],
            'bonus_text': pairing_result[7],
            'bonus_paid': pairing_result[8],
            'worker_id': pairing_result[9],
            'assignment_id': pairing_result[10],
            'run_id': pairing_result[11],
        }

    def _format_assignment(self, assign_result):
        return {
            'assignment_id': assign_result[0],
            'status': assign_result[1],
            'approve_time': assign_result[2],
            'worker_id': assign_result[3],
            'hit_id': assign_result[4],
        }

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
        c.execute('SELECT COUNT(*) FROM hits WHERE id = ?;', (id, ))
        is_new_hit = c.fetchone()[0] == 0
        if is_new_hit:
            c.execute('UPDATE runs SET created = created + 1 WHERE id = ?;',
                      (task_group_id, ))

        c.execute('REPLACE INTO hits VALUES (?,?,?,?,?,?,?);',
                  (id, expiration, status, assignments_pending,
                   assignments_available, assignments_complete, task_group_id))
        conn.commit()

    def log_worker_accept_assignment(self, worker_id, assignment_id, hit_id,
                                     task_group_id=None):
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        conn = self._get_connection()
        c = conn.cursor()

        # Ensure worker exists, mark the accepted assignment
        c.execute('SELECT COUNT(*) FROM workers WHERE id = ?;', (worker_id, ))
        has_worker = c.fetchone()[0] > 0
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
        c.execute('INSERT INTO pairings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                  (AssignState.STATUS_NONE, None, None, None, None, None, 0,
                   '', False, worker_id, assignment_id, task_group_id))
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
        c.execute('''UPDATE assignments SET status = ?, approve_time = ?
                     WHERE id = ?;''',
                  ('Reviewable', approve_time, assignment_id))

        # Increment worker completed
        c.execute('UPDATE workers SET completed = completed + 1 WHERE id = ?;',
                  (worker_id, ))

        # update the payment data status
        c.execute('''UPDATE pairings SET status = ?, task_end = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (complete_type, time.time(), worker_id, assignment_id))

        # Update run data to have another completed
        c.execute('UPDATE runs SET completed = completed + 1 WHERE id = ?;',
                  (task_group_id, ))
        conn.commit()

    def log_abandon_assignment(self, worker_id, assignment_id, approve_time,
                               disconnect_type, task_group_id=None):
        '''Note that an assignment was completed'''
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        conn = self._get_connection()
        c = conn.cursor()

        # Update assign data to reviewable
        c.execute('''UPDATE assignments SET status = ?, approve_time = ?
                     WHERE id = ?;''',
                  ('Reviewable', approve_time, assignment_id))

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
                  (task_group_id, ))
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
        c.execute('''UPDATE pairings SET bonus_amount = ?, bonus_text = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (amount, reason, worker_id, assignment_id))
        conn.commit()

    def log_bonus_paid(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE pairings SET bonus_paid = ?
                     WHERE worker_id = ? AND assignment_id = ?;''',
                  (True, worker_id, assignment_id))
        conn.commit()

    def log_approve_assignment(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ? WHERE id = ?;',
                  ('Approved', assignment_id))
        c.execute('''UPDATE workers SET approved = approved + 1
                     WHERE id = ?;''',
                  (worker_id, ))
        conn.commit()

    def log_reject_assignment(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ? WHERE id = ?;',
                  ('Rejected', assignment_id))
        c.execute('''UPDATE workers SET rejected = rejected + 1
                     WHERE id = ?;''',
                  (worker_id, ))
        conn.commit()

    def get_worker_data(self, worker_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM workers WHERE id = ?;", (worker_id, ))
        results = c.fetchone()
        if results is None:
            return None
        return {
            'worker_id': results[0],
            'accepted': results[1],
            'disconnected': results[2],
            'completed': results[3],
            'approved': results[4],
            'rejected': results[5],
        }

    def get_assignment_data(self, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM assignments WHERE id = ?;", (assignment_id, ))
        results = c.fetchone()
        if results is None:
            return None
        return self._format_assignment(results)

    def get_worker_assignment_pairing(self, worker_id, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""SELECT * FROM pairings WHERE worker_id = ?
                     AND assignment_id = ?;""", (worker_id, assignment_id, ))
        results = c.fetchone()
        if results is None:
            return None
        return self._format_pairing(results)

    def get_run_data(self, task_group_id):
        '''get the run data for the given task_group_id, return None if not
        found.
        '''
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM runs WHERE id = ?;", (task_group_id, ))
        results = c.fetchone()
        if results is None:
            return None
        return {
            'run_id': results[0],
            'created': results[1],
            'maximum': results[2],
            'completed': results[3],
            'failed': results[4],
        }

    def get_hit_data(self, hit_id):
        '''get the hit data for the given hit_id, return None if not'''
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM hits WHERE id = ?;", (hit_id, ))
        results = c.fetchone()
        if results is None:
            return None
        return {
            'hit_id': results[0],
            'expiration': results[1],
            'hit_status': results[2],
            'assignments_pending': results[3],
            'assignments_available': results[4],
            'assignments_complete': results[5],
            'run_id': results[6],
        }

    def get_pairings_for_assignment(self, assignment_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM pairings WHERE assignment_id = ?;",
                  (assignment_id, ))
        results = c.fetchall()
        if results is None:
            return None
        return [self._format_pairing(result) for result in results]

    def get_pairings_for_conversation(self, conversation_id,
                                      task_group_id=None):
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'

        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""SELECT * FROM pairings WHERE conversation_id = ?
                     AND run_id = ?;""", (conversation_id, task_group_id))
        results = c.fetchall()
        return [self._format_pairing(result) for result in results]

    def get_all_assignments_for_worker(self, worker_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM assignments WHERE worker_id = ?;",
                  (worker_id, ))
        results = c.fetchall()
        return [self._format_assignment(result) for result in results]

    def get_all_pairings_for_worker(self, worker_id):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM pairings WHERE worker_id = ?;", (worker_id, ))
        results = c.fetchall()
        return [self._format_pairing(result) for result in results]

    def get_all_task_assignments_for_worker(self, worker_id,
                                            task_group_id=None):
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""SELECT assignments.id, assignments.status,
                     assignments.approve_time, assignments.worker_id,
                     assignments.hit_id
                     FROM assignments
                     INNER JOIN hits on assignments.hit_id = hits.id
                     WHERE assignments.worker_id = ? AND hits.run_id = ?;""",
                  (worker_id, task_group_id))
        results = c.fetchall()
        return [self._format_assignment(result) for result in results]

    def get_all_task_pairings_for_worker(self, worker_id, task_group_id=None):
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM pairings WHERE worker_id = ? AND run_id = ?;",
                  (worker_id, task_group_id))
        results = c.fetchall()
        return [self._format_pairing(result) for result in results]
