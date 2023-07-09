# OpenAI Chat Completion API
This is an agent that interfaces with [OpenAI's chat completion api](https://platform.openai.com/docs/api-reference/chat/create) (/v1/chat/completions).

The chat completion endpoint supports the most advanced large language models as of June 2023 such as
* [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)
* [GPT-4](https://platform.openai.com/docs/models/gpt-4) 

## Setup
```bash
pip install openai
```

More info on setup is outlined in the official docs [here](https://platform.openai.com/docs/api-reference/introduction). 

Once the openai Python package is installed, you can start using the endpoint as long as you have a valid OpenAI API key generated and ready-to-use. 

##  Interactive example

```
parlai self_chat -m openai_chat_completions --num-self-chats 1 --selfchat-max-turns 6 --openai-api-key <insert your api key> --max-tokens 40 --model-name gpt-3.5-turbo --init-prompt "You are chatting with a fellow Wizard of the grand frost order. You are defending the village from sword wielding golden retrievers raiding the countryside." --name wizard1 --role user --counterparty-role user --counterparty-name moose --partner-model-file zoo:blender/blender_90M/model
```

### Output
```
[context]: Hi!
[OpenaiChatCompletionsAgent_1]: Hello fellow wizard! I see you're also busy defending the village from these sword-wielding golden retrievers. These creatures are quite strong and ferocious, aren't they?
   [TransformerGenerator_2]: i ' m not a wizard , but i ' ve always wanted to be one . what do you do for a living ?
[OpenaiChatCompletionsAgent_1]: Oh, I'm not sure if I would call it a living, per se. As a wizard of the grand frost order, my duty is to protect the village and its people from all sorts of
   [TransformerGenerator_2]: that ' s very noble of you . i ' d love to be a grand wizard one day .
[OpenaiChatCompletionsAgent_1]: It takes years of study and practice to become a grand wizard, but if you have the determination and passion for it, you can certainly achieve it. As a grand wizard, you won't just be
   [TransformerGenerator_2]: do you have any other hobbies besides being a wizard ? i like to think of it as a hobby .
[OpenaiChatCompletionsAgent_1]: As a wizard, my primary focus is on studying and practicing magic to improve my abilities and protect the village. However, when I have some free time, I enjoy reading books on history, mythology,
   [TransformerGenerator_2]: what kind of books do you like to read ? i have a lot of free time as well .
[OpenaiChatCompletionsAgent_1]: I enjoy reading books on history, folklore, and mythology. I find these topics fascinating, and they often give me inspiration for spells and incantations. I also enjoy reading fictional works, such as
   [TransformerGenerator_2]: i like fantasy books too . i like the ones that focus on the real world and not just fantasy .
```

## Self chat example
```
parlai interactive -m openai_chat_completions --openai-api-key <insert your api key> --max-tokens 40 --model-name gpt-4
```

### Output
```
Enter Your Message: Can you describe a pack of golden retriever knights roaming the countryside?
[OpenaiChatCompletionsAgent]: In the enchanting countryside, a majestic sight awaited anyone who happened to stumble upon it. A pack of Golden Retriever knights, glorious canines draped in gleaming armor, solemnly roamed
Enter Your Message: Can you write a sentence describing how they roam the land for name brand kibble?
[OpenaiChatCompletionsAgent]: the vast landscapes in pursuit of the fabled name-brand kibble, rumored to grant strength and power to those valiant enough to consume its heavenly morsels.
```

## Limitations
This API wrapper has three major limitations
1. Cost - Repeatedly prompting the API can be expensive. 
2. Rate limiting - API queries can run into rate limiting issues which will cause the conversation to error out. [Official docs](https://platform.openai.com/docs/guides/rate-limits) offers more insight on dealing with this issue.
3. Token Limit -  A combination of prompt and response can usually only be up to 8k tokens and may be smaller depending on the model requested for chat completions [official docs](https://openai.com/pricing). This limits the size of both the initial prompt as well as the length of conversation that we can feed back into the model. Exceeding this limit will cause the conversation to error out.
4. Self Chat - A self chat conducted between two OpenAI completion agents will not properly use the name and role arguments (as well as the counterparty versions). When this occurs, the turn history is not accurate because both agent-1 and agent-2 believe that their utterances are attached to `name` and `role` and that the other speaker is attributed to `counterparty-name` and `counterparty-role`. Ideally, agent-2 identifies its utterances to match `counterparty-name` and `counterparty-role`.
