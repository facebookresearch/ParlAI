from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import display_messages
from io import BytesIO

# from multiprocessing import Pool, Manager
import socketserver
import time

HOST_NAME = 'localhost'
PORT_NUMBER = 8080
SHARED = {}


class MyHandler(BaseHTTPRequestHandler):

    def interactive_running(self, opt, reply_text):
        reply = {}
        reply['text'] = reply_text
        model_res = SHARED['agent'].observe(reply)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        paths = {
            '/interact': {'status': 200},
        }
        if self.path == '/interact':
            # self.send_header('Content-type', 'text/plain')
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            model_response = self.interactive_running(SHARED.get('opt'), body.decode('utf-8'))

            self.send_response(200)
            self.end_headers()
            response = BytesIO()
            response.write(body)
            self.wfile.write(response.getvalue())

        else:
            self.respond({'status': 500})

    def do_GET(self):
        paths = {
            '/': {'status': 203},
            '/favicon.ico': {'status': 202},    # Need for chromes
        }
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path, text=None):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = '''
        <html><head><title>Title goes here.</title></head>
        <body><p>This is a test.</p>
        <form id = "interact">
            <input type="text" name="text" id="text">
            <br><input type="submit" value="Submit">
        </form>
        <p id="demo"></p>
        <script>
            document.getElementById("interact").addEventListener("submit", function(event){
                event.preventDefault()
                var text = document.getElementById("text").value;
                fetch('/interact', {
                    headers: {
                        'Content-Type': 'text/plain'
                    },
                    method: 'POST',
                    body: text
                })
                fetch('/interact', {
                    headers: {
                        'Content-Type': 'text/plain'
                    },
                    method: 'GET',
                })
            });
        </script>
        </body></html>
        '''
        return bytes(content, 'UTF-8')

    def respond(self, opts):
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)


def setup_interactive(shared):
    parser = setup_args()
    SHARED['opt'] = parser.parse_args(print_args=False)
    print_parser = parser

    if print_parser is not None:
        if print_parser is True and isinstance(self.server.SHARED['opt'], ParlaiParser):
            print_parser = SHARED['opt']
        elif print_parser is False:
            print_parser = None
    if isinstance(SHARED['opt'], ParlaiParser):
        print('[ Deprecated Warning: interactive should be passed opt not Parser ]')
        SHARED['opt'] = SHARED['opt'].parse_args()

    SHARED['opt']['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    SHARED['agent'] = create_agent(SHARED.get('opt'), requireModelExists=True)
    SHARED['world'] = create_task(SHARED.get('opt'), SHARED['agent'])

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = SHARED['agent'].opt
        print_parser.print_args()


if __name__ == '__main__':
    setup_interactive(SHARED)
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)

    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
