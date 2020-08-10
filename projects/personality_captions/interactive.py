#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Talk with a model using a web UI.
"""


from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.image_featurizers import ImageLoader
from typing import Dict, Any

import json
import cgi
import PIL.Image as Image
from base64 import b64decode
import io
import os

HOST_NAME = 'localhost'
PORT = 8080
SHARED: Dict[str, Any] = {}
IMAGE_LOADER = None
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
              <section class="hero is-info is-large has-background-light has-text-grey-dark">
                <div id="parent" class="hero-body">
                    <article class="media" id="photo-info">
                      <figure class="media-left">
                        <span class="icon is-large">
                          <i class="fas fa-robot fas fa-2x"></i>
                        </span>
                      </figure>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <img id="preview" src="Examples.png"/ style="max-height:300px">
                          </p>
                        </div>
                      </div>
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
                          Upload an image:
                          <input class="input" type="file" id="userInImage" accept="image/*">
                        </p>
                            Choose a personality:
                            <select name="pers_list" form="interact" id="pers_select">
                            </select>
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
            var personalities = ["Adventurous", "Appreciative (Grateful)", "Articulate (Well-spoken, Expressive)", "Attractive", "Brilliant", "Calm", "Captivating", "Caring", "Charming", "Cheerful", "Clever", "Colorful (Full of Life, Interesting)", "Compassionate (Sympathetic, Warm)", "Confident", "Considerate", "Contemplative (Reflective, Thoughtful)", "Courageous", "Creative", "Cultured (Refined, Educated)", "Curious", "Daring", "Deep", "Dramatic", "Earnest (Enthusiastic)", "Elegant", "Eloquent (Well-spoken, Expressive)", "Empathetic", "Energetic", "Enthusiastic", "Exciting", "Extraordinary", "Freethinking", "Fun-loving", "Gentle", "Happy", "Honest", "Humble", "Humorous", "Idealistic", "Imaginative", "Insightful", "Intelligent", "Kind", "Knowledgeable", "Logical", "Meticulous (Precise, Thorough)", "Objective (Detached, Impartial)", "Observant", "Open", "Optimistic", "Passionate", "Patriotic", "Peaceful", "Perceptive", "Playful", "Practical", "Profound", "Rational", "Realistic", "Reflective", "Relaxed", "Respectful", "Romantic", "Rustic (Rural)", "Scholarly", "Sensitive", "Sentimental", "Serious", "Simple", "Sophisticated", "Spirited", "Spontaneous", "Stoic (Unemotional, Matter-of-fact)", "Suave (Charming, Smooth)", "Sweet", "Sympathetic", "Vivacious (Lively, Animated)", "Warm", "Wise", "Witty", "Youthful", "Absentminded", "Aggressive", "Amusing", "Artful", "Boyish", "Breezy (Relaxed, Informal)", "Businesslike", "Casual", "Cerebral (Intellectual, Logical)", "Complex", "Conservative (Traditional, Conventional)", "Contradictory", "Cute", "Dreamy", "Dry", "Emotional", "Enigmatic (Cryptic, Obscure)", "Formal", "Glamorous", "High-spirited", "Impersonal", "Intense", "Maternal (Mother-like)", "Mellow (Soothing, Sweet)", "Mystical", "Neutral", "Old-fashioned", "Ordinary", "Questioning", "Sarcastic", "Sensual", "Skeptical", "Solemn", "Stylish", "Tough", "Whimsical (Playful, Fanciful)", "Abrasive (Annoying, Irritating)", "Airy (Casual, Not Serious)", "Aloof (Detached, Distant)", "Angry", "Anxious", "Apathetic (Uncaring, Disinterested)", "Argumentative", "Arrogant", "Artificial", "Assertive", "Barbaric", "Bewildered (Astonished, Confused)", "Bizarre", "Bland", "Blunt", "Boisterous (Rowdy, Loud)", "Childish", "Coarse (Not Fine, Crass)", "Cold", "Conceited (Arrogant, Egotistical)", "Confused", "Contemptible (Despicable, Vile)", "Cowardly", "Crazy", "Critical", "Cruel", "Cynical (Doubtful, Skeptical)", "Destructive", "Devious", "Discouraging", "Disturbing", "Dull", "Egocentric (Self-centered)", "Envious", "Erratic", "Escapist (Dreamer, Seeks Distraction)", "Excitable", "Extravagant", "Extreme", "Fanatical", "Fanciful", "Fatalistic (Bleak, Gloomy)", "Fawning (Flattering, Deferential)", "Fearful", "Fickle (Changeable, Temperamental)", "Fiery", "Foolish", "Frightening", "Frivolous (Trivial, Silly)", "Gloomy", "Grand", "Grim", "Hateful", "Haughty (Arrogant, Snobbish)", "Hostile", "Irrational", "Irritable", "Lazy", "Malicious", "Melancholic", "Miserable", "Money-minded", "Monstrous", "Moody", "Morbid", "Narcissistic (Self-centered, Egotistical)", "Neurotic (Manic, Obsessive)", "Nihilistic", "Obnoxious", "Obsessive", "Odd", "Offhand", "Opinionated", "Outrageous", "Overimaginative", "Paranoid", "Passive", "Pompous (Self-important, Arrogant)", "Pretentious (Snobbish, Showy)", "Provocative", "Quirky", "Resentful", "Ridiculous", "Rigid", "Rowdy", "Scornful", "Shy", "Silly", "Stiff", "Stupid", "Tense", "Uncreative", "Unimaginative", "Unrealistic", "Vacuous (Empty, Unintelligent)", "Vague", "Wishful", "Zany"];

            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"

                var figure = document.createElement("figure");
                figure.className = "media-left";

                var span = document.createElement("span");
                span.className = "icon is-large";

                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : "");

                var media = document.createElement("div");
                media.className = "media-content";

                var content = document.createElement("div");
                content.className = "content";

                var para = document.createElement("p");
                para.id = "model-response";
                var paraText = document.createTextNode(text);

                var strong = document.createElement("strong");
                strong.innerHTML = agent;
                var br = document.createElement("br");

                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(paraText);
                content.appendChild(para);
                media.appendChild(content);

                span.appendChild(icon);
                figure.appendChild(span);

                article.appendChild(figure);
                article.appendChild(media);

                return article;
            }}
            function fetchResult(personality, image_data) {{
                var formData = new FormData();
                formData.append('personality', personality);
                formData.append('image', image_data);
                fetch('/interact', {{

                    method: 'POST',
                    body: formData
                }}).then(response=>response.json()).then(data=>{{
                    var parDiv = document.getElementById("parent");

                    // Change info for Model response
                    var oldResponse = document.getElementById("model-response");
                    if (oldResponse) {{
                        oldResponse.parentNode.remove(oldResponse);
                    }}
                    parDiv.append(createChatRow("Model", data.text));
                    window.scrollTo(0,document.body.scrollHeight);
                }});

            }}
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()
                var personality = document.getElementById("pers_select").value;
                var img_input = document.getElementById("userInImage");
                var preview = document.getElementById("preview");
                if (img_input.files && img_input.files[0]) {{
                  var reader = new FileReader();
                  reader.onload = function (e) {{
                    preview.setAttribute('src', e.target.result);
                    fetchResult(personality, e.target.result);
                  }};
                  reader.readAsDataURL(img_input.files[0]);
                }} else {{
                  preview.setAttribute('src', 'Examples.png');
                }}
            }});
            var sel = document.getElementById("pers_select");
            for (var idx in personalities) {{
                personality = personalities[idx];
                var opt = document.createElement('option');
                opt.innerHTML = personality;
                opt.value = personality;
                sel.appendChild(opt);
            }}
        </script>

    </body>
</html>
"""  # noqa: E501


class MyHandler(BaseHTTPRequestHandler):
    """
    Interactive Handler.
    """

    def interactive_running(self, data):
        """
        Generate a model response.

        :param data:
            data to send to model

        :return:
            model act dictionary
        """
        reply = {}
        reply['text'] = data['personality'][0].decode()
        img_data = str(data['image'][0])
        _, encoded = img_data.split(',', 1)
        image = Image.open(io.BytesIO(b64decode(encoded))).convert('RGB')
        reply['image'] = SHARED['image_loader'].extract(image)
        SHARED['agent'].observe(reply)
        model_res = SHARED['agent'].act()
        return model_res

    def do_HEAD(self):
        """
        Handle headers.
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        """
        Handle POST.
        """
        if self.path != '/interact':
            return self.respond({'status': 500})
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        postvars = cgi.parse_multipart(self.rfile, pdict)
        model_response = self.interactive_running(postvars)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        json_str = json.dumps(model_response)
        self.wfile.write(bytes(json_str, 'utf-8'))

    def do_GET(self):
        """
        Handle GET.
        """
        paths = {
            '/': {'status': 200},
            '/favicon.ico': {'status': 202},  # Need for chrome
        }
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path, text=None):
        """
        Generate HTTP.
        """
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
        return bytes(content, 'UTF-8')

    def respond(self, opts):
        """
        Respond.
        """
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)


def setup_interactive():
    """
    Set up the interactive script.
    """
    parser = setup_args()
    opt = parser.parse_args()
    if not opt.get('model_file'):
        raise RuntimeError('Please specify a model file')
    if opt.get('fixed_cands_path') is None:
        opt['fixed_cands_path'] = os.path.join(
            '/'.join(opt.get('model_file').split('/')[:-1]), 'candidates.txt'
        )
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    opt['image_mode'] = 'resnet152'
    SHARED['opt'] = opt
    SHARED['image_loader'] = ImageLoader(opt)

    # Create model and assign it to the specified task
    SHARED['agent'] = create_agent(opt, requireModelExists=True)
    SHARED['world'] = create_task(opt, SHARED['agent'])


if __name__ == '__main__':
    setup_interactive()
    server_class = HTTPServer
    Handler = MyHandler
    Handler.protocol_version = 'HTTP/1.0'
    httpd = server_class((HOST_NAME, PORT), Handler)
    print('http://{}:{}/'.format(HOST_NAME, PORT))

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
