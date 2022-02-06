# Flask API demo

__Authors__: Khushal Jethava


## Parl.ai model implement on flask framework

With this script, you can implement your fine-tune or pretrained parlai model very easily.


### Example Code

```python

from flask import Flask, request

from parlai.core.agents import create_agent_from_model_file


app = Flask(__name__)

def Model_init():
  # import model from the model file can be pretrained or fine tuned

  blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")
  return blender_agent

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

````

### How to

Just change the model path and model name inside the create_agent_from_model_file function.

```python
blender_agent = create_agent_from_model_file("path_to_model_or_zoo_name")
```

Now Run the script and start the flask webserver

Pass the message format defined below:

```json
{
    "UserText" : "Your Inputted Text"

}
```

It will generate below response

```json
{
    "response": "Model Response"
}
```