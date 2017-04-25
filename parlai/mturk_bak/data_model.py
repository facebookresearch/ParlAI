# Copyright 2004-present Facebook. All Rights Reserved.
import os
import sys
import time
import json
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, UnicodeText, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine, func

is_python_2 = False
if sys.version_info[0] < 3:
    is_python_2 = True

 
Base = declarative_base()
engine = None
session = None


class Message(Base):
    __tablename__ = 'message'
    id = Column(Integer, primary_key=True)
    task_group_id = Column(String(255), index=True)  # We assign a new task_group_id for each HIT group
    conversation_id = Column(Integer, index=True)
    agent_id = Column(String(255))
    message_content = Column(UnicodeText)
    created_time = Column(TIMESTAMP, server_default=func.now(), index=True)

 
def init_database(host, db_name, username, password):
    # Create an engine
    engine = create_engine('postgres://'+username+':'+password+'@'+host+':5432/'+db_name)
     
    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine

    session_maker = sessionmaker(bind=engine)

    return engine, session_maker


def send_new_message(db_session, task_group_id, conversation_id, agent_id, message_text=None, reward=None, done=False, binary_file_bytes=None, binary_file_type=None):
    """
    Message format:
    {
        # ParlAI observation/action dict fields:
        "text": xxx, # text of speaker(s)
        "id": xxx, # id of speaker(s)
        "reward": xxx,
        "done": xxx, # signals end of episode

        # Extra fields for MTurk state maintenance
        "message_id": xxx, # populated with record on database
        "timestamp": xxx, # populated with record on database
    }
    """

    # ParlAI observation/action dict fields:
    new_message = {
        "text": message_text,
        "id": agent_id,
    }
    if reward:
        new_message['reward'] = reward    
    new_message['done'] = done

    message_content = json.dumps(new_message)
    if is_python_2:
        message_content = unicode(message_content)

    new_message_object = Message(
        task_group_id = task_group_id,
        conversation_id = conversation_id,
        agent_id = agent_id,
        message_content = message_content
    )
    db_session.add(new_message_object)
    db_session.commit()

    return new_message_object


def get_new_messages(db_session, task_group_id, conversation_id=None, after_message_id=None, excluded_agent_id=None, populate_state_info=False):
    """
    Return:
    conversation_dict = {
        <conversation_id>: [
            {
                # ParlAI observation/action dict fields:
                "text": xxx, # text of speaker(s)
                "id": xxx, # id of speaker(s)
                "reward": xxx,
                "done": xxx, # signals end of episode

                # Extra fields for MTurk state maintenance
                "message_id": xxx, # populated with record on database
                "timestamp": xxx, # populated with record on database
            }
        ], ...
    },
    last_message_id
    """

    if not after_message_id:
        after_message_id = -1

    excluded_agent_ids = []
    if excluded_agent_id:
        excluded_agent_ids = [excluded_agent_id]

    last_message_id = None

    query = db_session.query(Message).filter(Message.task_group_id==task_group_id).filter(~Message.agent_id.in_(excluded_agent_ids)).filter(Message.id > after_message_id)
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
        new_message_dict['done'] = message_content.get('done', False)

        if populate_state_info:
            new_message_dict['message_id'] = new_message_object.id
            new_message_dict['timestamp'] = time.mktime(new_message_object.created_time.timetuple()) + new_message_object.created_time.microsecond * 1e-6
        
        if conversation_id not in conversation_dict:
            conversation_dict[conversation_id] = []
        conversation_dict[conversation_id].append(new_message_dict)

    return conversation_dict, last_message_id
