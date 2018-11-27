# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""ParlAI Server file"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import importlib
import inspect
import json
import logging
import os
import time
import traceback

import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape


from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.mturk_manager import MTurkManager

DEFAULT_PORT = 8095
DEFAULT_HOSTNAME = "localhost"
DEFAULT_DB_FILE = 'pmt_data.db'
DEFAULT_SB_DB_FILE = 'pmt_sbdata.db'
IS_DEBUG = True

here = os.path.abspath(os.path.dirname(__file__))


def row_to_dict(row):
    return (dict(zip(row.keys(), row)))


def get_rand_id():
    return str(hex(int(time.time() * 10000000))[2:])


def force_dir(path):
    """Make sure the parent dir exists for path so we can write a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_path(filename):
    """Get the path to an asset."""
    cwd = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(cwd, filename)


tornado_settings = {
    "autoescape": None,
    "debug": IS_DEBUG,
    "static_path": get_path('static'),
    "template_path": get_path('static'),
    "compiled_template_cache": False
}


class Application(tornado.web.Application):
    def __init__(self, port=DEFAULT_PORT, db_file=DEFAULT_DB_FILE,
                 is_sandbox=False):
        self.state = {'is_sandbox': is_sandbox}
        self.subs = {}
        self.sources = {}
        self.port = port
        self.data_handler = MTurkDataHandler(file_name=db_file)
        self.mturk_manager = MTurkManager.make_taskless_instance(is_sandbox)
        self.mturk_manager.db_logger = self.data_handler

        # TODO load some state from DB

        handlers = [
            (r"/app/(.*)", AppHandler, {'app': self}),
            (r"/tasks", TaskListHandler, {'app': self}),
            (r"/workers", WorkerListHandler, {'app': self}),
            (r"/runs/(.*)", RunHandler, {'app': self}),
            (r"/workers/(.*)", WorkerHandler, {'app': self}),
            (r"/assignments/(.*)", AssignmentHandler, {'app': self}),
            (r"/approve/(.*)", ApprovalHandler, {'app': self}),
            (r"/reject/(.*)", RejectionHandler, {'app': self}),
            (r"/reverse_rejection/(.*)", ReverseHandler, {'app': self}),
            (r"/block/(.*)", BlockHandler, {'app': self}),
            (r"/bonus/(.*)", BonusHandler, {'app': self}),
            (r"/error/(.*)", ErrorHandler, {'app': self}),
            (r"/socket", SocketHandler, {'app': self}),
            (r"/", RedirectHandler),
        ]
        super(Application, self).__init__(handlers, **tornado_settings)


def broadcast_msg(handler, msg, target_subs=None):
    if target_subs is None:
        target_subs = handler.subs.values()
    for sub in target_subs:
        sub.write_message(json.dumps(msg))


def send_to_sources(handler, msg):
    target_sources = handler.sources.values()
    for source in target_sources:
        source.write_message(json.dumps(msg))


class SocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, app):
        self.port = app.port
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources

    def check_origin(self, origin):
        return True

    def broadcast_default(self, target_subs=None):
        # TODO create any default content that needs to go in msg
        # if target_subs is None:
        #     target_subs = self.subs.values()
        # broadcast_msg(self, msg, target_subs)
        pass

    def open(self):
        self.sid = get_rand_id()
        if self not in list(self.subs.values()):
            self.subs[self.sid] = self
        logging.info(
            'Opened new socket from ip: {}'.format(self.request.remote_ip))

        self.write_message(
            json.dumps({'command': 'register', 'data': self.sid}))
        self.broadcast_default([self])

    def on_message(self, message):
        logging.info('from web client: {}'.format(message))
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))

        cmd = msg.get('cmd')

        # TODO flesh out with stubs as they develop
        if cmd == 'todo':
            # Do something
            pass

    def on_close(self):
        if self in list(self.subs.values()):
            self.subs.pop(self.sid, None)


class BaseHandler(tornado.web.RequestHandler):
    def __init__(self, *request, **kwargs):
        self.include_host = False
        super(BaseHandler, self).__init__(*request, **kwargs)

    def write_error(self, status_code, **kwargs):
        logging.error("ERROR: %s: %s" % (status_code, kwargs))
        if self.settings.get("debug") and "exc_info" in kwargs:
            logging.error("rendering error page")
            exc_info = kwargs["exc_info"]
            # exc_info is a tuple consisting of:
            # 1. The class of the Exception
            # 2. The actual Exception that was thrown
            # 3. The traceback opbject
            try:
                params = {
                    'error': exc_info[1],
                    'trace_info': traceback.format_exception(*exc_info),
                    'request': self.request.__dict__
                }

                self.render("error.html", **params)
                logging.error("rendering complete")
            except Exception as e:
                logging.error(e)


class AppHandler(tornado.web.RequestHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, args):
        self.render('index.html', initial_location=args)
        return


class RedirectHandler(tornado.web.RequestHandler):
    def post(self):
        self.set_status(404)

    def get(self):
        print('redirecting')
        self.redirect('/app/tasks')


class TaskListHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def post(self):
        req = tornado.escape.json_decode(
            tornado.escape.to_basestring(self.request.body)
        )
        self.write(json.dumps({'t': 'testing!', 'req': req}))

    def get(self):
        results = self.data_handler.get_all_run_data()
        processed_results = []
        for res in results:
            processed_results.append(row_to_dict(res))
        for result in processed_results:
            # TODO implemenent
            result['run_status'] = 'unimplemented'

        self.write(json.dumps(processed_results))


def merge_assignments_with_pairings(assignments, pairings, log_id):
    processed_assignments = {}
    for res in assignments:
        assign_dict = row_to_dict(res)
        processed_assignments[assign_dict['assignment_id']] = assign_dict
    for res in pairings:
        pairing_dict = row_to_dict(res)
        assign_id = pairing_dict['assignment_id']
        if assign_id not in processed_assignments:
            print('assignment {} missing from assign table for {}'
                  ''.format(assign_id, log_id))
        pairing_dict['world_status'] = pairing_dict['status']
        del pairing_dict['status']
        processed_assignments[assign_id].update(pairing_dict)
    return list(processed_assignments.values())


class RunHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, task_target):
        hits = self.data_handler.get_hits_for_run(task_target)
        processed_hits = []
        for res in hits:
            processed_hits.append(row_to_dict(res))
        assignments = self.data_handler.get_assignments_for_run(task_target)
        pairings = self.data_handler.get_pairings_for_run(task_target)
        processed_assignments = merge_assignments_with_pairings(
            assignments, pairings, 'task {}'.format(task_target))
        run_details = row_to_dict(self.data_handler.get_run_data(task_target))
        # TODO implement run status determination
        run_details['run_status'] = 'unimplemented'
        data = {
            'run_details': run_details,
            'assignments': processed_assignments,
            'hits': processed_hits,
        }

        self.write(json.dumps(data))


class WorkerListHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self):
        results = self.data_handler.get_all_worker_data()
        processed_results = []
        for res in results:
            processed_results.append(row_to_dict(res))

        self.write(json.dumps(processed_results))


class WorkerHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, worker_target):
        assignments = self.data_handler.get_all_assignments_for_worker(
            worker_target)
        pairings = self.data_handler.get_all_pairings_for_worker(
            worker_target)
        processed_assignments = merge_assignments_with_pairings(
            assignments, pairings, 'task {}'.format(worker_target))
        worker_details = row_to_dict(
            self.data_handler.get_worker_data(worker_target))
        data = {
            'worker_details': worker_details,
            'assignments': processed_assignments,
        }

        self.write(json.dumps(data))


class AssignmentHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, assignment_target):
        # Extract assignment
        assignments = [self.data_handler.get_assignment_data(
            assignment_target)]
        pairings = self.data_handler.get_pairings_for_assignment(
            assignment_target)
        processed_assignments = merge_assignments_with_pairings(
            assignments, pairings, 'assignment {}'.format(assignment_target))
        assignment = processed_assignments[0]

        # Get assignment details to retrieve assignment content
        run_id = assignment['run_id']
        onboarding_id = assignment['onboarding_id']
        conversation_id = assignment['conversation_id']
        worker_id = assignment['worker_id']

        onboard_data = None
        if onboarding_id is not None:
            onboard_data = MTurkDataHandler.get_conversation_data(
                run_id, onboarding_id, worker_id, self.state['is_sandbox'])

        assignment_content = {
            'onboarding': onboard_data,
            'task': MTurkDataHandler.get_conversation_data(
                run_id, conversation_id, worker_id, self.state['is_sandbox']),
        }

        # Get assignment instruction html
        taskname = '_'.join(run_id.split('_')[:-1])
        find_location = 'parlai.mturk.tasks.{}.task_config'.format(taskname)
        find_location_internal = \
            'parlai_internal.mturk.tasks.{}.task_config'.format(taskname)
        try:
            # Try to find the task config in public tasks
            t = importlib.import_module(find_location)
            task_instructions = t.task_config['task_description']
        except ImportError:
            try:
                # Try to find the task in local tasks
                t = importlib.import_module(find_location_internal)
                task_instructions = t.task_config['task_description']
            except ImportError:
                task_instructions = None

        data = {
            'assignment_details': assignment,
            'assignment_content': assignment_content,
            'assignment_instructions': task_instructions,
        }

        self.write(json.dumps(data))


class ApprovalHandler(BaseHandler):
    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, assignment_target):
        self.mturk_manager.approve_work(assignment_target)
        data = {
            'status': True,
        }
        self.write(json.dumps(data))


class ReverseHandler(BaseHandler):
    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, assignment_target):
        self.mturk_manager.approve_work(assignment_target, True)
        print('Assignment {} had rejection reversed'.format(assignment_target))
        data = {
            'status': True,
        }
        self.write(json.dumps(data))


class RejectionHandler(BaseHandler):
    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, assignment_target):
        data = tornado.escape.json_decode(self.request.body)
        reason = data['reason']
        self.mturk_manager.reject_work(assignment_target, reason)
        print('Rejected {} for reason {}'.format(assignment_target, reason))
        data = {
            'status': True,
        }
        self.write(json.dumps(data))


class BlockHandler(BaseHandler):
    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, worker_target):
        data = tornado.escape.json_decode(self.request.body)
        reason = data['reason']
        self.mturk_manager.block_worker(worker_target, reason)
        print('Blocked {} for reason {}'.format(worker_target, reason))
        data = {
            'status': True,
        }
        self.write(json.dumps(data))


class BonusHandler(BaseHandler):
    def initialize(self, app):
        self.mturk_manager = app.mturk_manager

    def post(self, worker_target):
        """Requests to /bonus/{worker_id} will give a bonus to that worker.
        Requires a reason, assignment_id, a unique token (for idempotence),
        and the bonus amount IN CENTS
        """
        data = tornado.escape.json_decode(self.request.body)
        reason = data['reason']
        assignment_id = data['assignment_id']
        bonus_cents = data['bonus_cents']
        token = data['bonus_token']

        dollar_amount = bonus_cents / 100.0
        self.mturk_manager.pay_bonus(
            worker_target, dollar_amount, assignment_id, reason, token)
        print('Bonused ${} to {} for reason {}'.format(
            dollar_amount, worker_target, reason))
        data = {
            'status': True,
        }
        self.write(json.dumps(data))


class ErrorHandler(BaseHandler):
    def get(self, text):
        error_text = text or "test error"
        raise Exception(error_text)


def start_server(port=DEFAULT_PORT, hostname=DEFAULT_HOSTNAME,
                 db_file=DEFAULT_DB_FILE, is_sandbox=False):
    print("It's Alive!")
    app = Application(port=port, db_file=db_file, is_sandbox=is_sandbox)
    app.listen(port, max_buffer_size=1024 ** 3)
    logging.info("Application Started")

    if "HOSTNAME" in os.environ and hostname == DEFAULT_HOSTNAME:
        hostname = os.environ["HOSTNAME"]
    else:
        hostname = hostname
    print("You can navigate to http://%s:%s" % (hostname, port))
    tornado.ioloop.IOLoop.current().start()


def main():
    parser = argparse.ArgumentParser(
        description='Start the ParlAI-MTurk task managing server.')
    parser.add_argument('--port', metavar='port', type=int,
                        default=DEFAULT_PORT,
                        help='port to run the server on.')
    parser.add_argument('--hostname', metavar='hostname', type=str,
                        default=DEFAULT_HOSTNAME,
                        help='host to run the server on.')
    parser.add_argument('--sandbox', dest='sandbox',
                        action='store_true', default=False,
                        help='Run the server using sandbox data')
    parser.add_argument('--db_file', metavar='db_file', type=str,
                        default=DEFAULT_DB_FILE,
                        help='name of database to use (in core/run_data)')
    parser.add_argument('--logging_level', metavar='logger_level',
                        default='INFO',
                        help='logging level (default = INFO). Can take logging'
                             ' level name or int (example: 20)')
    FLAGS = parser.parse_args()

    if FLAGS.sandbox:
        if FLAGS.db_file == DEFAULT_DB_FILE:
            FLAGS.db_file = DEFAULT_SB_DB_FILE

    logging_level = logging._checkLevel(FLAGS.logging_level)
    logging.getLogger().setLevel(logging_level)

    start_server(port=FLAGS.port, hostname=FLAGS.hostname,
                 db_file=FLAGS.db_file, is_sandbox=FLAGS.sandbox)


if __name__ == "__main__":
    main()
