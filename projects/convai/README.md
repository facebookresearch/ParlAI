# convai-testing-system integration example

[ConvAI](http://convai.io) is a competition of chatbots, where people talking to chatbots or other people. 
The aim of this competition is to create ranking of bots from dummy bot to human and help researchers find way to 
human-level conversational intelligence.

[convai-testing-system](https://github.com/deepmipt/convai-testing-system) is software developed to provide backend for ConvAI challenge. Here you find code for integration of `convai-testing-system` with Parl.AI, so you can use Parl.AI for participation in ConvAI or connect to your own instance of testing system.

For your agent to work with the `convai-testing-system` instance for ConvAI challenge you need:

- request a token by email from [info@convai.io](info@convai.io)
- implement your own agent basing on provided [`convai_bot.py`](./convai_bot.py)

To run your agent for ConvAI challenge by default: `python convai_bot.py -bi <BOT_TOKEN> -rbu https://ipavlov.mipt.ru/nipsrouter/`
