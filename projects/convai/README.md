# ConvAI 'wild' evaluation integration

[ConvAI](http://convai.io) is a competition of chatbots, during the competition people are talking to chatbots or other people. 
The aim of this competition is to create a ranking of bots from dummy bot to human and help researchers find a way to 
human-level conversational intelligence.

[ConvAI Router Bot](https://github.com/deepmipt/convai_router_bot) is a software developed to provide backend for ConvAI challenge. Here you find code for integration of `convai_router_bot` with Parl.AI, so you can use Parl.AI for participation in ConvAI or connect to your own instance of testing system.

For your agent to work with the `convai_router_bot` instance for ConvAI challenge you need:

- request a token by email from [info@convai.io](info@convai.io)
- implement your own agent basing on provided [`convai_bot.py`](./convai_bot.py)

To run your agent for ConvAI challenge by default: `python convai_bot.py -bi <BOT_TOKEN> -rbu <SERVER_URL>`
