# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import time
import json
import math
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, UnicodeText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, func
from sqlalchemy.pool import NullPool
from sqlalchemy import inspect
 
Base = declarative_base()
engine = None

COMMAND_GET_NEW_MESSAGES = 'COMMAND_GET_NEW_MESSAGES' # MTurk agent is expected to get new messages from server
COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE' # MTurk agent is expected to send a new message to server
COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT' # MTurk agent is expected to hit "DONE" button and submit the HIT

def object_as_dict(obj):
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}

class Message(Base):
    __tablename__ = 'message'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)  # We assign a new task_group_id for each HIT group
    conversation_id = Column(String(255), index=True)
    sender_agent_id = Column(String(255), index=True)
    receiver_agent_id = Column(String(255), index=True, default=None)
    message_content = Column(UnicodeText)

class Command(Base):
    __tablename__ = 'command'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)
    conversation_id = Column(String(255), index=True)
    receiver_agent_id = Column(String(255), index=True)
    command = Column(String(255))

class MTurkHITAgentAllocation(Base):
    __tablename__ = 'mturk_hit_agent_allocation'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)
    agent_id = Column(String(255), index=True)
    conversation_id = Column(String(255), index=True, default=None)
    assignment_id = Column(String(255), default=None)
    hit_id = Column(String(255), default=None)
    worker_id = Column(String(255), default=None)


def check_database_health():
    session_maker = sessionmaker(bind=engine)
    session = scoped_session(session_maker)

    try:
        # Check whether all tables exist
        for model_class in [Message, MTurkHITAgentAllocation]:
            if not engine.dialect.has_table(engine, model_class.__tablename__):
                return 'missing_table'

        # Try insert new objects with current schema
        try:
            test_message = Message(id=0, task_group_id='Test', conversation_id='Test', sender_agent_id='Test', receiver_agent_id='Test', message_content='Test')
            session.add(test_message)
            session.commit()
            session.delete(test_message)
            session.commit()

            test_command = Command(id=0, task_group_id='Test', conversation_id='Test', receiver_agent_id='Test', command='Test')
            session.add(test_command)
            session.commit()
            session.delete(test_command)
            session.commit()

            test_agent_allocation = MTurkHITAgentAllocation(id=0, task_group_id='Test', agent_id='Test', conversation_id='Test', assignment_id='Test', hit_id='Test', worker_id='Test')
            session.add(test_agent_allocation)
            session.commit()
            session.delete(test_agent_allocation)
            session.commit()

            return 'healthy'
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return 'inconsistent_schema'
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise e
        return 'unknown_error'


def setup_database_engine(host, db_name, username, password):
    # Create an engine
    global engine
    engine = create_engine('postgres://'+username+':'+password+'@'+host+':5432/'+db_name, poolclass=NullPool)


def close_connection(db_engine, db_session):
    db_session.close()
    db_engine.dispose()


def init_database():
    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine

    session_maker = sessionmaker(bind=engine)

    return engine, session_maker


def send_new_command(db_session, task_group_id, conversation_id, receiver_agent_id, command):
    new_command_object = Command(
        task_group_id = task_group_id,
        conversation_id = conversation_id,
        receiver_agent_id = receiver_agent_id,
        command = command
    )
    db_session.add(new_command_object)
    db_session.commit()

    return new_command_object


def get_command(db_session, task_group_id, conversation_id, receiver_agent_id, after_command_id):
    query = db_session.query(Command).filter(Command.task_group_id==task_group_id) \
                                    .filter(Command.conversation_id==conversation_id) \
                                    .filter(Command.receiver_agent_id==receiver_agent_id) \
                                    .filter(Command.id > after_command_id) \
                                    .order_by(Command.id)
    command_object = query.first()
    if command_object:
        return command_object
    else:
        return None


def send_new_message(db_session, task_group_id, conversation_id, sender_agent_id, receiver_agent_id, message_text=None, reward=None, episode_done=False):
    """
    Message format:
    {
        # ParlAI observation/action dict fields:
        "text": xxx, # text of speaker(s)
        "id": xxx, # id of speaker(s)
        "reward": xxx,
        "episode_done": xxx, # signals end of episode
    }
    """

    # ParlAI observation/action dict fields:
    new_message = {
        "text": message_text,
        "id": sender_agent_id,
    }
    if reward:
        new_message['reward'] = reward
    new_message['episode_done'] = episode_done

    message_content = json.dumps(new_message)
    try:
        message_content = unicode(message_content)
    except NameError:   # unicode() was removed in Python 3
        pass

    new_message_object = Message(
        task_group_id = task_group_id,
        conversation_id = conversation_id,
        sender_agent_id = sender_agent_id,
        receiver_agent_id = receiver_agent_id,
        message_content = message_content
    )
    db_session.add(new_message_object)
    db_session.commit()

    return new_message_object


def get_new_messages(db_session, task_group_id, receiver_agent_id, conversation_id=None, after_message_id=None, excluded_sender_agent_id=None, included_sender_agent_id=None, populate_meta_info=False):
    """
    Return:
    conversation_dict = {
        <conversation_id>: [
            {
                # ParlAI observation/action dict fields:
                "text": xxx, # text of speaker(s)
                "id": xxx, # id of speaker(s)
                "reward": xxx,
                "episode_done": xxx, # signals end of episode

                # Extra fields for MTurk state maintenance
                "message_id": xxx, # populated with record on database
            }
        ], ...
    },
    last_message_id
    """

    if not after_message_id:
        after_message_id = -1

    included_sender_agent_ids = []
    if included_sender_agent_id:
        included_sender_agent_ids = [included_sender_agent_id]

    excluded_sender_agent_ids = []
    if excluded_sender_agent_id:
        excluded_sender_agent_ids = [excluded_sender_agent_id]

    last_message_id = None

    query = db_session.query(Message).filter(Message.task_group_id==task_group_id).filter(Message.id > after_message_id)
    if len(included_sender_agent_ids) > 0:
        query = query.filter(Message.sender_agent_id.in_(included_sender_agent_ids))
    if len(excluded_sender_agent_ids) > 0:
        query = query.filter(~Message.sender_agent_id.in_(excluded_sender_agent_ids))
    if conversation_id:
        query = query.filter(Message.conversation_id==conversation_id)
    if receiver_agent_id:
        query = query.filter(Message.receiver_agent_id==receiver_agent_id)
    new_message_objects = query.order_by(Message.id)
    conversation_dict = {}

    for new_message_object in new_message_objects:
        conversation_id = new_message_object.conversation_id
        message_content = json.loads(new_message_object.message_content)
        text = message_content['text']

        if not last_message_id or new_message_object.id > last_message_id:
            last_message_id = new_message_object.id

        new_message_dict = {
            "text": text,
            "id": new_message_object.sender_agent_id,
        }
        if 'reward' in message_content:
            new_message_dict['reward'] = message_content['reward']
        new_message_dict['episode_done'] = message_content.get('episode_done', False)
        new_message_dict['receiver_agent_id'] = new_message_object.receiver_agent_id

        if populate_meta_info:
            new_message_dict['message_id'] = new_message_object.id
            
        if conversation_id not in conversation_dict:
            conversation_dict[conversation_id] = []
        conversation_dict[conversation_id].append(new_message_dict)

    return conversation_dict, last_message_id


def sync_hit_assignment_info(db_session, task_group_id, agent_id, num_assignments, assignment_id, hit_id, worker_id):
    new_allocation_object = MTurkHITAgentAllocation(
                                task_group_id=task_group_id,
                                agent_id=agent_id,
                                conversation_id=None,
                                assignment_id=assignment_id,
                                hit_id=hit_id,
                                worker_id=worker_id
                            )
    db_session.add(new_allocation_object)
    db_session.commit()

    object_id = new_allocation_object.id
    existing_allocation_id_list = db_session.query(MTurkHITAgentAllocation.id) \
                                    .filter(MTurkHITAgentAllocation.task_group_id==task_group_id) \
                                    .filter(MTurkHITAgentAllocation.agent_id==agent_id) \
                                    .order_by(MTurkHITAgentAllocation.id).all()
    existing_allocation_id_list = [id for (id, ) in existing_allocation_id_list]
    index_in_list = existing_allocation_id_list.index(object_id)

    hit_index = int(math.floor(index_in_list / num_assignments) + 1)
    assignment_index = index_in_list % num_assignments + 1
    conversation_id = str(hit_index) + '_' + str(assignment_index)
    new_allocation_object.conversation_id = conversation_id
    db_session.add(new_allocation_object)
    db_session.commit()

    return {'hit_index': hit_index, 'assignment_index': assignment_index}

def get_hit_assignment_info(db_session, task_group_id, agent_id, conversation_id):
    existing_allocation_object = db_session.query(MTurkHITAgentAllocation) \
                                .filter(MTurkHITAgentAllocation.task_group_id==task_group_id) \
                                .filter(MTurkHITAgentAllocation.agent_id==agent_id) \
                                .filter(MTurkHITAgentAllocation.conversation_id==conversation_id) \
                                .first()
    assignment_id = None
    hit_id = None
    worker_id = None
    if existing_allocation_object:
        assignment_id = existing_allocation_object.assignment_id
        hit_id = existing_allocation_object.hit_id
        worker_id = existing_allocation_object.worker_id
    return {
        'assignment_id': assignment_id,
        'hit_id': hit_id,
        'worker_id': worker_id
    }
