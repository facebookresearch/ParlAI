#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from flask import Flask, request
from parlai.core.agents import create_agent_from_model_file


app = Flask(__name__)


@app.route("/response", methods=("GET", "POST"))
def chatbot_response():
    data = request.json
    blender_agent.observe({'text': data["UserText"], 'episode_done': False})
    response = blender_agent.act()
    return {'response': response['text']}


# main driver function
if __name__ == "__main__":
    blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
    app.run()
