from flask import Flask, render_template, request
from parlai.core.agents import create_agent_from_model_file


app = Flask(__name__)

def Model_init():
  #import model from the model file can be pretrained or fine tuned
  blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
  return blender_agent

#blender_agent = index()
@app.route("/response", methods=["GET","POST"])
def chatbot_response():
    data = request.json
    blender_agent.observe({'text': data["UserText"], 'episode_done': False})
    response = blender_agent.act()
    return {"response" : response['text']} 


# main driver function
if __name__ == "__main__":
  blender_agent = Model_init()
  app.run()
