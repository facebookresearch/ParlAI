import requests
from time import sleep

API_URI = "http://84.201.158.229:8080/api/send_message"
FILENAME = "dialog_output.txt"


def send_message(message_text, message_history):
    response = requests.post(API_URI, json={"message_text": message_text, "message_history": message_history})
    return response.json()["text"]


def print_to_file_decorator(func):
    def new_func(*args, **kwargs):
        func(*args, **kwargs)
        with open(FILENAME, "a") as f:
            f.write(' '.join(list(map(str, args))) + "\n")
    return new_func


class BotClient:
    def __init__(self, message_history=None):
        self.message_history = message_history if message_history else []

    def send_message(self, message):
        response = send_message(message, self.message_history)
        self.message_history.append(message)
        self.message_history.append(response)

        return response


class BotConversation:
    def __init__(self, first_bot: BotClient, second_bot: BotClient):
        self.first_bot = first_bot
        self.second_bot = second_bot

        self.initial_message = "Hello!"
        self.message_timeout = 2

    def start(self):
        print(f"Bot 2: {self.initial_message}")
        self.second_bot.message_history.append(self.initial_message)

        first_response = self.first_bot.send_message(self.initial_message)
        print(f"Bot 1: {first_response}")

        while True:
            second_response = self.second_bot.send_message(first_response)
            print(f"Bot 2: {second_response}")

            sleep(self.message_timeout)

            first_response = self.first_bot.send_message(second_response)
            print(f"Bot 1: {first_response}")

            sleep(self.message_timeout)


open(FILENAME, "w")
print = print_to_file_decorator(print)

FirstBot = BotClient(["your persona: i hate you shit fuck you i am 23 and i work as photographer"])
SecondBot = BotClient(["your persona: i hate you you fucking bitch i am 17 and i am in school my hobby is airplanes"])

print("# INITIAL CONTEXTS #")
print("Bot 1:", FirstBot.message_history)
print("Bot 2:", SecondBot.message_history)
print()
print("# DIALOG #")

MainBotConversation = BotConversation(FirstBot, SecondBot)
MainBotConversation.start()
