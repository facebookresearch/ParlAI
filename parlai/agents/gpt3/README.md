# GPT3
This is an agent that interfaces with [OpenAI's completion V1 api](https://platform.openai.com/docs/api-reference/completions) (/v1/completions). The completions v1 API broadly covers the models that fall under GPT-3 like:
* text-davinci-003
* text-curie-001
* text-ada-001

The model works by prompting the completion API repeatedly with the proper chat history appended to the prompt.

A more comprehensive set of available models or engines is listed in the official docs [here](https://platform.openai.com/docs/models/model-endpoint-compatibility)

We've written the model wrapper such that it can handle both:
1. A proper chat history: by appending turns of conversation to the final prompt
2. An initial conversation prompt which can offer instruction to the GPT-3 model 

## Setup
```bash
pip install openai
```

More info on setup is outlined in the official docs [here](https://platform.openai.com/docs/api-reference/introduction). 

Once the openai Python package is installed, you can start using the endpoint as long as you have a valid OpenAI API key generated and ready-to-use. 

##  Interactive example

```
parlai interactive -m gpt3 --openai-api-key <insert your api key> --max-tokens 40 --model-name text-ada-001
```

## Self chat example
```
parlai self_chat -m gpt3 --num-self-chats 1 --selfchat-max-turns 5 --openai-api-key <insert your api key> --max-tokens 40 --model-name text-davinci-002 --partner-model-file zoo:blender/blender_90M/model
```

## Limitations
This API wrapper has three major limitations
1. Cost - Repeatedly prompting the completion API can be expensive especially on the more expensive models like Davinci. 
2. Rate limiting - API queries can run into rate limiting issues which will cause the conversation to error out. [Official docs](https://platform.openai.com/docs/guides/rate-limits) offers more insight on dealing with this issue.
3. Token Limit -  A combination of prompt and response can usually only be up to 2049 and may be smaller depending on the model for GPT-3 [official docs](https://platform.openai.com/docs/models/gpt-3). This limits the size of both the initial prompt as well as the length of conversation that we can feed back into the model. Exceeding this limit will cause the conversation to error out.
