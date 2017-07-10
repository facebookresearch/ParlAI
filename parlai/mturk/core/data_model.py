# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import sys
import time
import json
import math
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, UnicodeText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, func
from sqlalchemy.pool import NullPool

is_python_2 = False
if sys.version_info[0] < 3:
    is_python_2 = True
 
Base = declarative_base()
engine = None


class Message(Base):
    __tablename__ = 'message'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)  # We assign a new task_group_id for each HIT group
    conversation_id = Column(String(255), index=True)
    agent_id = Column(String(255), index=True)
    assignment_id = Column(String(255), default=None)
    hit_id = Column(String(255), default=None)
    worker_id = Column(String(255), default=None)
    message_content = Column(UnicodeText)

class MTurkHITAgentAllocation(Base):
    __tablename__ = 'mturk_hit_agent_allocation'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)
    agent_id = Column(String(255), index=True)


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
            test_message = Message(id=0, task_group_id='Test', conversation_id='Test', agent_id='Test', assignment_id='Test', hit_id='Test', worker_id='Test', message_content='Test')
            session.add(test_message)
            session.commit()
            session.delete(test_message)
            session.commit()

            test_agent_allocation = MTurkHITAgentAllocation(id=0, task_group_id='Test', agent_id='Test')
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


def send_new_message(db_session, task_group_id, conversation_id, agent_id, message_text=None, reward=None, episode_done=False, assignment_id=None, hit_id=None, worker_id=None):
    """
    Message format:
    {
        # ParlAI observation/action dict fields:
        "text": xxx, # text of speaker(s)
        "id": xxx, # id of speaker(s)
        "reward": xxx,
        "episode_done": xxx, # signals end of episode

        # Only from MTurk worker
        "assignment_id": xxx,
        "hit_id": xxx,
        "worker_id": xxx,
    }
    """

    # ParlAI observation/action dict fields:
    new_message = {
        "text": message_text,
        "id": agent_id,
    }
    if reward:
        new_message['reward'] = reward
    new_message['episode_done'] = episode_done

    message_content = json.dumps(new_message)
    if is_python_2:
        message_content = unicode(message_content)

    new_message_object = Message(
        task_group_id = task_group_id,
        conversation_id = conversation_id,
        agent_id = agent_id,
        assignment_id = assignment_id,
        hit_id = hit_id,
        worker_id = worker_id,
        message_content = message_content
    )
    db_session.add(new_message_object)
    db_session.commit()

    return new_message_object


def send_new_messages_in_bulk(db_session, new_messages):
    """
    Same message format as in send_new_message(), but also needs to include the following in each message:
    task_group_id
    conversation_id

    NOTE: we don't need returned objects if we are sending new messages from the local ParlAI system, so no return for this method is fine
    """

    new_message_objects = []
    for new_message in new_messages:
        new_message_dict = {
            "text": new_message['text'],
            "id": new_message['id'],
        }
        if 'reward' in new_message:
            new_message_dict['reward'] = new_message['reward']
        new_message_dict['episode_done'] = new_message.get('episode_done', False)
        message_content = json.dumps(new_message_dict)
        if is_python_2:
            message_content = unicode(message_content)
        new_message_objects.append(Message(
            task_group_id = new_message['task_group_id'],
            conversation_id = new_message['conversation_id'],
            agent_id = new_message['id'],
            message_content = message_content
        ))

    db_session.bulk_save_objects(new_message_objects)
    db_session.commit()


def get_new_messages(db_session, task_group_id, conversation_id=None, after_message_id=None, excluded_agent_id=None, included_agent_id=None, populate_meta_info=False, populate_hit_info=False):
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

    included_agent_ids = []
    if included_agent_id:
        included_agent_ids = [included_agent_id]

    excluded_agent_ids = []
    if excluded_agent_id:
        excluded_agent_ids = [excluded_agent_id]

    last_message_id = None

    query = db_session.query(Message).filter(Message.task_group_id==task_group_id).filter(Message.id > after_message_id)
    if len(included_agent_ids) > 0:
        query = query.filter(Message.agent_id.in_(included_agent_ids))
    if len(excluded_agent_ids) > 0:
        query = query.filter(~Message.agent_id.in_(excluded_agent_ids))
    if conversation_id:
        query = query.filter(Message.conversation_id==conversation_id)
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
            "id": new_message_object.agent_id,
        }
        if 'reward' in message_content:
            new_message_dict['reward'] = message_content['reward']
        new_message_dict['episode_done'] = message_content.get('episode_done', False)

        if populate_meta_info:
            new_message_dict['message_id'] = new_message_object.id

        if populate_hit_info:
            new_message_dict['assignment_id'] = new_message_object.assignment_id
            new_message_dict['hit_id'] = new_message_object.hit_id
            new_message_dict['worker_id'] = new_message_object.worker_id
            
        if conversation_id not in conversation_dict:
            conversation_dict[conversation_id] = []
        conversation_dict[conversation_id].append(new_message_dict)

    return conversation_dict, last_message_id


def get_hit_index_and_assignment_index(db_session, task_group_id, agent_id, num_assignments):
    new_allocation_object = MTurkHITAgentAllocation(task_group_id=task_group_id, agent_id=agent_id)
    db_session.add(new_allocation_object)
    db_session.commit()
    object_id = new_allocation_object.id
    existing_allocation_id_list = db_session.query(MTurkHITAgentAllocation.id) \
                                    .filter(MTurkHITAgentAllocation.task_group_id==task_group_id) \
                                    .filter(MTurkHITAgentAllocation.agent_id==agent_id) \
                                    .order_by(MTurkHITAgentAllocation.id).all()
    existing_allocation_id_list = [id for (id, ) in existing_allocation_id_list]
    index_in_list = existing_allocation_id_list.index(object_id)
    return {'hit_index': math.floor(index_in_list / num_assignments) + 1, 'assignment_index': index_in_list % num_assignments + 1}
