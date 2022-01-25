#import required libraries
from flask import Flask, render_template, request
from parlai.core.agents import create_agent_from_model_file


# Flask constructor takes the name of
#initialize flask app
app = Flask(__name__)

#import model from the model file can be pretrained or fine tuned
blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")


# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route("/response", methods=["GET","POST"]) # API URL 
def chatbot_response(): # function name 
    data = request.json # Take input as json format	
    blender_agent.observe({'text': data["UserText"], 'episode_done': False}) #Give User inputted text to model
    response = blender_agent.act() #take response from the model
    return {"response" : response['text']} #return model response


# main driver function
if __name__ == "__main__":
    app.run()
