## Browser Chat
This allows you to participate in a ParlAI world as an agent using a local browser.
This extends the `websocket` chat service implementation to run a server locally,
which you can send and receive messages using a browser.

## Setup
1. Run:

 `python parlai/chat_service/services/browser_chat/run.py --config-path path/to/config.yml --port PORT_NUMBER`

  Example:

   `python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config.yml --port 10001`

2. Run: `python client.py --port PORT_NUMBER`
3. Interact

If no port number is specified in `--port` then the default port used will be `34596`. If specifying, ensure both port numbers match on client and server side.

## Note

If your 8080 port is already in use, please edit the port number in `_run_browser` in `client.py`.
