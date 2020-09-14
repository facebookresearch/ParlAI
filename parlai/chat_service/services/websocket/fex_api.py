import flask
import requests
import argparse
import json
import websockets
import uuid
import asyncio
import logging
import re
import threading
from flask import Flask, request, jsonify

logging.basicConfig(filename='parlai_fex_api.log', level=30)

parser = argparse.ArgumentParser(description="Simple API for ParlAI chat bot")
parser.add_argument('--hostname', default="localhost", help="ParlAI web server hostname.")
parser.add_argument('--port', type=int, default=8080, help="ParlAI web server port.")
parser.add_argument('--serving_hostname', default="0.0.0.0", help="API web server hostname.")
parser.add_argument('--serving_port', type=int, default=8080, help="API web server port.")

args = parser.parse_args()

hostname = args.hostname
port = args.port	
serving_hostname = args.serving_hostname
serving_port = args.serving_port

app = Flask(__name__)
blueprint = flask.Blueprint('parlai_api', __name__, template_folder='templates')

connections = {}
FEX_API_URI = "http://35.193.147.253:5665/gpt2_roleplay_api/v1/actions/get_response"

running = False

requests_queue = []
responses = {}


def get_random_id():
    return str(uuid.uuid4())


def format_message(message):
    while match := re.search("\s'\s", message):
        message = message[:match.start()] + "'" + message[match.end():]
        
    while match := re.search('\s[.?!,;:\']', message):
        message = message[:match.start()] + message[match.end() - 1:]

    while match := re.search('[.?!]\s[a-z]', message):
        message = message[:match.end() - 1] + message[match.end() - 1].capitalize() + message[match.end():]

    message = message[0].capitalize() + message[1:]

    return message


class ParlaiAPI:
    @staticmethod
    def parse():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            if not requests_queue:
                continue
            request = requests_queue.pop(0)
            result = loop.run_until_complete(request[1]())
            responses[request[0]] = result
    
    @staticmethod
    async def send_message(user_message, message_history=[], persona=False):
        new_history = []
        first_message = None
        context = []

        for i, message in enumerate(message_history):
            if i == 0 and message.startswith('your persona:'):
                first_message = message[13:].strip()
                continue
            if message.startswith('your persona:'):
                continue
            new_history.append(message)

        if first_message:
            context.append(first_message)

        context += new_history[-17:]
        context.append(user_message)

        response = requests.post(FEX_API_URI, json={"context": context}).json()

        try:
            response_text = format_message(response["response"])
        except Exception as e:
            return response

        return {"text": response_text}


@blueprint.route('/api/send_message', methods=["POST"])
def send_message():
    request_id = get_random_id()
    data = request.get_json()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    message_text, message_history = data.get('message_text', None), data.get('message_history', [])

    requests_queue.append([request_id,
                     lambda: ParlaiAPI.send_message(message_text, message_history)])
    logging.warning(str(requests))
    while request_id not in responses:
        pass

    result = responses[request_id]
    del responses[request_id]
    return result


@blueprint.route('/api/start_conversation', methods=["POST"])
def start_conversation():
    # DEPRECATED

    return {'text': 'Hello!'}


@blueprint.route('/api/end_conversation', methods=["POST"])
def end_conversation():
    # DEPRECATED

    return {'text': 'Goodbye!'}



async def main():
    thread = threading.Thread(target=ParlaiAPI.parse)
    thread.start()
    app.register_blueprint(blueprint)
    app.run(host=serving_hostname, port=serving_port)

main_loop = asyncio.get_event_loop()
main_loop.run_until_complete(main())
