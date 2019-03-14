from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import display_messages

import json
import socketserver
import time

HOST_NAME = 'localhost'
PORT_NUMBER = 8080
SHARED = {}
STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark	">
                <div id="parent" class="hero-body">
                    <article class="media">
                      <figure class="media-left">
                        <span class="icon is-large">
                          <i class="fas fa-robot fas fa-2x"></i>
                        </span>
                      </figure>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Model</strong>
                            <br>
                            Enter a message, and the model will respond interactively.
                          </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth">
                  <form id = "interact">
                      <div class="field is-grouped">
                        <p class="control is-expanded">
                          <input class="input" type="text" id="userIn" placeholder="Type in a message">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Submit
                          </button>
                        </p>
                      </div>
                  </form>
                </div>
              </section>
            </div>
        </div>

        <script>
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()
                var text = document.getElementById("userIn").value;
                document.getElementById('userIn').value = "";

                fetch('/interact', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: text
                }}).then(response=>response.json()).then(data=>{{
                    var parDiv = document.getElementById("parent");

                    var article = document.createElement("article");
                    article.className = "media"

                    var figure = document.createElement("figure");
                    figure.className = "media-left";

                    var span = document.createElement("span");
                    span.className = "icon is-large";

                    var icon = document.createElement("i");
                    icon.className = "fas fa-user fas fa-2x";

                    var media = document.createElement("div");
                    media.className = "media-content";

                    var content = document.createElement("div");
                    content.className = "content";

                    var para = document.createElement("p");
                    var paraText = document.createTextNode(text);

                    var strong = document.createElement("strong");
                    strong.innerHTML = "You";
                    var br = document.createElement("br");

                    para.appendChild(strong);
                    para.appendChild(br);
                    para.appendChild(paraText);
                    console.log("paraText")
                    console.log(paraText);
                    content.appendChild(para);
                    media.appendChild(content);

                    span.appendChild(icon);
                    figure.appendChild(span);

                    article.appendChild(figure);
                    article.appendChild(media);

                    parDiv.append(article);

                    // Change info for Model response

                    var articleM = document.createElement("article");
                    articleM.className = "media";

                    var figureM = document.createElement("figure");
                    figureM.className = "media-left";

                    var spanM = document.createElement("span");
                    spanM.className = "icon is-large";

                    var iconM = document.createElement("i");
                    iconM.className = "fas fa-robot fas fa-2x";

                    var mediaM = document.createElement("div");
                    mediaM.className = "media-content";

                    var contentM = document.createElement("div");
                    contentM.className = "content";

                    var paraM = document.createElement("p");
                    var paraTextM = document.createTextNode(data.text);
                    console.log('paraTextM');
                    console.log(paraTextM);

                    var strongM = document.createElement("strong");
                    strongM.innerHTML = "Model";
                    var brM = document.createElement("br");

                    paraM.appendChild(strongM);
                    paraM.appendChild(brM);
                    paraM.appendChild(paraTextM);
                    contentM.appendChild(paraM);
                    mediaM.appendChild(contentM);

                    spanM.appendChild(iconM);
                    figureM.appendChild(spanM);

                    articleM.appendChild(figureM);
                    articleM.appendChild(mediaM);

                    parDiv.append(articleM);
                }})
            }});
        </script>

    </body>
</html>
"""

class MyHandler(BaseHTTPRequestHandler):

    def interactive_running(self, opt, reply_text):
        reply = {}
        reply['text'] = reply_text
        SHARED['agent'].observe(reply)
        model_res = SHARED['agent'].act()
        return model_res

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        if self.path == '/interact':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            model_response = self.interactive_running(SHARED.get('opt'), body.decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_str = json.dumps(model_response)
            self.wfile.write(bytes(json_str, 'utf-8'))

        else:
            self.respond({'status': 500})

    def do_GET(self):
        paths = {
            '/': {'status': 200},
            '/favicon.ico': {'status': 202},    # Need for chrome
        }
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path, text=None):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = WEB_HTML.format(
            STYLE_SHEET,
            FONT_AWESOME,
        )
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
    Handler = MyHandler
    Handler.protocol_version = 'HTTP/1.0'
    httpd = server_class((HOST_NAME, PORT_NUMBER), Handler)

    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
