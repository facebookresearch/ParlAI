import flask
import requests
import argparse
import json
import websockets
import uuid
import asyncio
import logging
import re
from flask import Flask, request, jsonify

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
websocket_uri = f"ws://{hostname}:{port}/websocket"

running = False


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
    async def send_message(user_message, message_history=[], persona=False):
        if persona:
            message = "your persona: "
        else:
            message = ""

        message += user_message

        request_dict = {"text": message, "message_history": message_history}
        request_string = json.dumps(request_dict)
        request_bytes = bytes(request_string, encoding="UTF-8")
        print(request_bytes)
        
        try:
            async with websockets.connect(websocket_uri) as ws:
                await ws.send(request_bytes)

                response = await ws.recv()

                response = json.loads(response)
                print(response)

                try:
                    response['text'] = format_message(response['text'])
                except Exception as e:
                    print(e)

                return response
        except Exception as e:
            return {'text': str(e)}
         


@blueprint.route('/api/send_message', methods=["POST"])
def send_message():
    global running
    while running:
        pass
    running = True
    data = request.get_json()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    message_text, message_history = data.get('message_text', None), data.get('message_history', [])

    result = loop.run_until_complete(ParlaiAPI.send_message(message_text, message_history))
    running = False
    return result


@blueprint.route('/api/send_person_message', methods=["POST"])
def send_person_message():
    global running
    while running:
        pass
    running = True
    data = request.get_json()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    message_text, message_history = data.get('message_text', None), data.get('message_history', [])

    result = loop.run_until_complete(ParlaiAPI.send_message(message_text, message_history, persona=True))
    running = False
    return result


@blueprint.route('/api/start_conversation', methods=["GET", "POST"])
def start_conversation():
    data = request.get_json()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    result = loop.run_until_complete(ParlaiAPI.send_message('begin'))
    return {'status': 'OK'}


@blueprint.route('/api/end_conversation', methods=["GET", "POST"])
def end_conversation():
    data = request.get_json()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    result = loop.run_until_complete(ParlaiAPI.send_message(user_id, '[DONE]'))
    return {'status': 'OK'}


async def main():
    app.register_blueprint(blueprint)
    app.run(host=serving_hostname, port=serving_port)

main_loop = asyncio.get_event_loop()
main_loop.run_until_complete(main())
