## Flask API demo

__Authors__: Khushal Jethava


### Parl.ai model implement on flask framework

With this script, you can implement your fine-tune or pretrained parlai model very easily.

### How to

Just change the model path and model name inside the create_agent_from_model_file function.

```python
blender_agent = create_agent_from_model_file("Model Name")
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