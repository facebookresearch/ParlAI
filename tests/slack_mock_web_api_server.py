"""
From https://github.com/slackapi/python-slack-sdk/blob/main/tests/rtm/mock_web_api_server.py
"""
import json
import logging
import threading
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Type
from unittest import TestCase


class MockHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    default_request_version = "HTTP/1.1"
    logger = logging.getLogger(__name__)

    def is_valid_token(self):
        return "authorization" in self.headers and str(
            self.headers["authorization"]
        ).startswith("Bearer xoxb-")

    def is_invalid_rtm_start(self):
        return (
            "authorization" in self.headers
            and str(self.headers["authorization"]).startswith("Bearer xoxb-rtm.start")
            and str(self.path) != "/rtm.start"
        )

    def set_common_headers(self):
        self.send_header("content-type", "application/json;charset=utf-8")
        self.send_header("connection", "close")
        self.end_headers()

    rtm_start_success = {
        "ok": True,
        "url": "ws://localhost:8765",
        "self": {"id": "U01234ABC", "name": "robotoverlord"},
        "team": {
            "domain": "exampledomain",
            "id": "T123450FP",
            "name": "ExampleName",
        },
    }

    rtm_start_failure = {
        "ok": False,
        "error": "invalid_auth",
    }

    def _handle(self):
        if self.is_invalid_rtm_start():
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.set_common_headers()
            return

        self.send_response(HTTPStatus.OK)
        self.set_common_headers()
        body = (
            self.rtm_start_success if self.is_valid_token() else self.rtm_start_failure
        )
        self.wfile.write(json.dumps(body).encode("utf-8"))
        self.wfile.close()

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()


class MockServerThread(threading.Thread):
    def __init__(
        self, test: TestCase, handler: Type[SimpleHTTPRequestHandler] = MockHandler
    ):
        threading.Thread.__init__(self)
        self.handler = handler
        self.test = test

    def run(self):
        self.server = HTTPServer(("localhost", 8888), self.handler)
        self.test.server_url = "http://localhost:8888"
        self.test.host, self.test.port = self.server.socket.getsockname()
        self.test.server_started.set()  # threading.Event()

        self.test = None
        try:
            self.server.serve_forever(0.05)
        finally:
            self.server.server_close()

    def stop(self):
        self.server.shutdown()
        self.join()


def setup_mock_web_api_server(test: TestCase):
    test.server_started = threading.Event()
    test.thread = MockServerThread(test)
    test.thread.start()
    test.server_started.wait()


def cleanup_mock_web_api_server(test: TestCase):
    test.thread.stop()
    test.thread = None
