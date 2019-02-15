# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
import threading
import traceback
import asyncio
import sh
import shlex
import shutil
import subprocess
import uuid

import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape

from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.webapp.run_mocks.mock_turk_manager import MockTurkManager
from parlai import __path__ as parlai_path
parlai_path = parlai_path[0]

try:
    from parlai_internal import __path__ as parlai_int_path
    parlai_int_path = parlai_int_path[0]
except Exception:
    parlai_int_path = None


DEFAULT_PORT = 8095
DEFAULT_HOSTNAME = "localhost"
DEFAULT_DB_FILE = 'pmt_data.db'
DEFAULT_SB_DB_FILE = 'pmt_sbdata.db'
IS_DEBUG = True

here = os.path.abspath(os.path.dirname(__file__))

tasks = {}


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


def get_mturk_task_module(task_name, target_module):
    full_location = tasks[task_name]
    guess_loc = full_location.split('tasks/')[1]
    guess_class = '.'.join(guess_loc.split('/'))
    task_module = target_module.format(guess_class)
    base_format = 'parlai.mturk.tasks.{}'
    if 'parlai_internal' in full_location:
        base_format = 'parlai_internal.mturk.tasks.{}'
    find_location = base_format.format(task_module)
    # Try to find the task at specified location
    return importlib.import_module(find_location)


def get_run_module(task_name):
    return get_mturk_task_module(task_name, '{}.run')


def get_config_module(task_name):
    return get_mturk_task_module(task_name, '{}.task_config')


tornado_settings = {
    "autoescape": None,
    "debug": IS_DEBUG,
    "static_path": get_path('static'),
    "template_path": get_path('static'),
    "compiled_template_cache": False
}


class PacketWrap(object):
    def __init__(self, data):
        self.data = data
        self.id = None


class Application(tornado.web.Application):
    def __init__(self, port=DEFAULT_PORT, db_file=DEFAULT_DB_FILE,
                 is_sandbox=False):
        self.state = {'is_sandbox': is_sandbox}
        self.subs = {}
        self.sources = {}
        self.port = port
        self.data_handler = MTurkDataHandler(file_name=db_file)
        self.manager = None  # MTurk manager for demo tasks
        self.mturk_manager = MTurkManager.make_taskless_instance(is_sandbox)
        self.mturk_manager.db_logger = self.data_handler
        self.task_manager = None  # This is the task mturk manager

        # TODO load some state from DB

        handlers = [
            (r"/app/(.*)", AppHandler, {'app': self}),
            (r"/run_list", RunListHandler, {'app': self}),
            (r"/task_list", TaskListHandler, {'app': self}),
            (r"/run_task/(.*)", TaskRunHandler, {'app': self}),
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
            (r"/socket", TaskSocketHandler, {'app': self}),
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
        self.redirect('/app/home')


class RunListHandler(BaseHandler):
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


class TaskListHandler(BaseHandler):
    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self):
        results = {t: {'task_name': t, 'dir': v} for t, v in tasks.items()}
        for task_name, directory in tasks.items():
            results[task_name]['internal'] = 'parlai_internal' in directory
            results[task_name]['has_custom'] = False
            results[task_name]['react_frontend'] = False
            try:
                config = get_config_module(task_name)
                frontend_version = config.task_config.get('frontend_version')
                if (frontend_version is not None and frontend_version >= 1):
                    results[task_name]['react_frontend'] = True
                if os.path.isfile(os.path.join(
                        directory, 'frontend', 'components', 'custom.jsx')):
                    results[task_name]['has_custom'] = True
            except Exception as e:
                print('Exception {} when loading task details for {}'.format(
                    e, task_name
                ))
                pass
            results[task_name]['active_runs'] = 'unimplemented'  # TODO
            results[task_name]['all_runs'] = 'unimplemented'  # TODO

        self.write(json.dumps(list(results.values())))


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

        workers = set()
        # get feedback data and put into assignments if present
        for assignment in processed_assignments:
            assignment['received_feedback'] = None
            run_id = assignment['run_id']
            conversation_id = assignment['conversation_id']
            worker_id = assignment['worker_id']
            workers.add(worker_id)
            if conversation_id is not None:
                task_data = MTurkDataHandler.get_conversation_data(
                    run_id, conversation_id, worker_id,
                    self.state['is_sandbox'])
                if task_data['data'] is not None:
                    assignment['received_feedback'] = \
                        task_data['data'].get('received_feedback')

        worker_data = {}
        for worker in workers:
            worker_data[worker] = \
                row_to_dict(self.data_handler.get_worker_data(worker))

        run_details = row_to_dict(self.data_handler.get_run_data(task_target))
        # TODO implement run status determination
        run_details['run_status'] = 'unimplemented'
        data = {
            'run_details': run_details,
            'worker_details': worker_data,
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
            'task_name': '_'.join(run_id.split('_')[:-1]),
        }

        # Get assignment instruction html. This can be much improved
        task_name = '_'.join(run_id.split('_')[:-1])
        try:
            # Try to find the task at specified location
            t = get_config_module(task_name)
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


class TaskSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, app):
        self.app = app
        self.sources = app.sources

    def check_origin(self, origin):
        return True

    def _run_socket(self):
        time.sleep(2)
        asyncio.set_event_loop(asyncio.new_event_loop())
        while self.alive and self.app.task_manager is not None:
            try:
                self.write_message(json.dumps({
                    'data': [agent.get_update_packet()
                             for agent in self.app.task_manager.agents],
                    'command': 'sync'
                }))
                time.sleep(0.2)
            except tornado.websocket.WebSocketClosedError:
                self.alive = False
                self.app.task_manager.timeout_all_agents()

    def open(self):
        self.sid = str(hex(int(time.time() * 10000000))[2:])
        self.alive = True
        if self not in list(self.sources.values()):
            self.sources[self.sid] = self
        logging.info('Opened task socket from ip: {}'.format(
            self.request.remote_ip))

        self.write_message(
            json.dumps({'command': 'alive', 'data': 'socket_alive'}))

        t = threading.Thread(target=self._run_socket)
        t.start()

    def on_message(self, message):
        logging.info('from frontend client: {}'.format(message))
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))
        message = msg['text']
        task_data = msg['task_data']
        sender_id = msg['sender']
        agent_id = msg['id']
        act = {
            'id': agent_id,
            'task_data': task_data,
            'text': message,
            'message_id': str(uuid.uuid4()),
        }
        t = threading.Thread(
            target=self.app.task_manager.on_new_message,
            args=(sender_id, PacketWrap(act)),
            daemon=True)
        t.start()

    def on_close(self):
        self.alive = False
        if self in list(self.sources.values()):
            self.sources.pop(self.sid, None)


class TaskRunHandler(BaseHandler):
    def initialize(self, app):
        self.app = app

    def post(self, task_target):
        """Requests to /run_task/{task_id} will launch a task locally
        for the given task. It will die after 20 mins if it doesn't end
        on its own.
        """
        try:
            # Load the run and task_config modules from the expected locations
            t = get_run_module(task_target)
            conf = get_config_module(task_target)
            # Set the run file's MTurkManager module to be MockTurkManager
            t.MTurkManager = MockTurkManager
            MockTurkManager.current_manager = None

            # Start a task thread, then wait for the task to be running
            task_thread = threading.Thread(target=t.main, name='Demo-Thread')
            task_thread.start()
            while MockTurkManager.current_manager is None:
                time.sleep(1)
            time.sleep(1)
            manager = MockTurkManager.current_manager

            # Register the current manager, then alive the agents
            self.app.task_manager = manager
            for agent in manager.agents:
                manager.worker_alive(
                    agent.worker_id, agent.hit_id, agent.assignment_id)

            # Tell frontend we're done, and give the initial packets.
            data = {
                'started': True,
                'data': [agent.get_update_packet() for agent in manager.agents],
                'task_config': conf.task_config,
            }
            self.write(json.dumps(data))
        except Exception as e:
            data = {
                'error': e,
            }
            print(repr(e))
            print(traceback.format_exc())
            self.write(json.dumps(data))


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


def crawl_dir_for_tasks(search_dir):
    found_dirs = {}  # taskname -> task directory
    contents = os.listdir(search_dir)
    for sub_dir in contents:
        full_sub_dir = os.path.join(search_dir, sub_dir)
        if os.path.exists(os.path.join(full_sub_dir, 'run.py')):
            # assume this is a task
            found_dirs[sub_dir] = full_sub_dir
        if os.path.isdir(full_sub_dir):
            found_dirs.update(crawl_dir_for_tasks(full_sub_dir))
    return found_dirs


def rebuild_source():
    # TODO move task parsing to its own helper file
    # Copy files to the correct locations,
    if os.path.exists('dev/task_components'):
        print('removing old build')
        sh.rm(shlex.split('-rf {}'.format('dev/task_components')))
    os.mkdir('dev/task_components')

    copy_dirs = {}
    parlai_task_dir = os.path.join(parlai_path, 'mturk', 'tasks')
    tasks.update(crawl_dir_for_tasks(parlai_task_dir))

    if parlai_int_path is not None:
        parlai_internal_task_dir = \
            os.path.join(parlai_int_path, 'mturk', 'tasks')
        tasks.update(crawl_dir_for_tasks(parlai_internal_task_dir))

    for task, task_dir in tasks.items():
        if os.path.exists(os.path.join(task_dir, 'frontend')):
            copy_dirs[task] = os.path.join(task_dir, 'frontend')

    for task, task_dir in copy_dirs.items():
        was_built = False
        if 'package.json' in os.listdir(task_dir):
            # need to build this component first
            os.chdir(task_dir)
            packages_installed = subprocess.call(['npm', 'install'])
            if packages_installed != 0:
                raise Exception('please make sure npm is installed, otherwise '
                                'view the above error for more info.')

            webpack_complete = subprocess.call(['npm', 'run', 'dev'])
            if webpack_complete != 0:
                raise Exception('Webpack appears to have failed to build your '
                                'frontend. See the above error for more '
                                'information.')
            was_built = True

        os.chdir(here)
        output_dir = os.path.join('dev/task_components', task)
        shutil.copytree(task_dir, output_dir)
        # need to move built custom.jsx back into components
        if was_built:
            shutil.copy2(os.path.join(output_dir, 'dist', 'custom.jsx'),
                         os.path.join(output_dir, 'components', 'custom.jsx'))
        # Need to give a copy of core_components to have same dir
        shutil.copy2(os.path.join(parlai_path, 'mturk', 'core', 'react_server',
                                  'dev', 'components', 'core_components.jsx'),
                     os.path.join(output_dir, 'components'))

    # Grab up to date version of core components
    shutil.copy2(os.path.join(parlai_path, 'mturk', 'core', 'react_server',
                              'dev', 'components', 'core_components.jsx'),
                 'dev/task_components')

    # build the full react server
    packages_installed = subprocess.call(['npm', 'install'])
    if packages_installed != 0:
        raise Exception('please make sure npm is installed, otherwise view '
                        'the above error for more info.')

    webpack_complete = subprocess.call(['npm', 'run', 'dev'])
    if webpack_complete != 0:
        raise Exception('Webpack appears to have failed to build your '
                        'frontend. See the above error for more information.')


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

    rebuild_source()

    start_server(port=FLAGS.port, hostname=FLAGS.hostname,
                 db_file=FLAGS.db_file, is_sandbox=FLAGS.sandbox)


if __name__ == "__main__":
    main()
