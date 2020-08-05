import requests
import argparse
from telegram.ext import Updater, MessageHandler, Filters
from telegram.ext import CallbackContext, CommandHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

TOKEN = "1156352622:AAEo8fqFYKZet_jpcCW2SlnYWQRp-PzQGxw"
translate_api_key = "trnsl.1.1.20181209T174121Z.671341eadeba8a97.dd5c1bcb8805bac6f9d977de43957e40e413b483"

parser = argparse.ArgumentParser(description="Russian Telegram bot for API testing.")
parser.add_argument('--api_hostname', default="localhost", help="ParlAI API hostname.")
parser.add_argument('--api_port', type=int, default=8080, help="ParlAI API port.")

args = parser.parse_args()

api_hostname = args.api_hostname
api_port = args.api_port
api_uri = f"http://{api_hostname}:{api_port}/api"


def translate(text, lang):
    response = requests.get('https://translate.yandex.net/api/v1.5/tr.json/translate',
                            params={'key': translate_api_key,
                                    'text': text,
                                    'lang': lang})
    response = response.json()
    return response['text'][0]


def send_response(update, context, response):
    quick_replies = response.get('quick_replies')
    text = response.get('text')

    text = translate(text, 'en-ru')

    if quick_replies:
        keyboard = [quick_replies]
        markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)

        update.message.reply_text(text, reply_markup=markup)

        return
    update.message.reply_text(text, reply_markup=ReplyKeyboardRemove())


def send_message(update, context):
    chat_id = update.message.chat_id
    message_text = translate(update.message.text, 'ru-en')

    response = requests.post(f'{api_uri}/send_message',
                             json={"message_text": message_text,
                                   "user_id": chat_id})

    try:
        response = response.json()
        send_response(update, context, response)
    except Exception as e:
        update.message.reply_text("Простите, мы не можем обработать ваш запрос. Попробуйте чуть позже.")
        raise e


def send_person_message(update, context):
    chat_id = update.message.chat_id
    message_text = ' '.join(list(map(str, context.args)))
    message_text = translate(message_text, 'ru-en')

    response = requests.post(f'{api_uri}/send_person_message',
                             json={"message_text": message_text,
                                   "user_id": chat_id})

    try:
        response = response.json()
        send_response(update, context, response)
    except Exception as e:
        update.message.reply_text("Простите, мы не можем обработать ваш запрос. Попробуйте чуть позже.")
        raise e


def start_conversation(update, context):
    chat_id = update.message.chat_id

    response = requests.post(f'{api_uri}/start_conversation',
                             json={"user_id": chat_id})

    try:
        response = response.json()
        send_response(update, context, response)
    except Exception as e:
        update.message.reply_text("Простите, мы не можем обработать ваш запрос. Попробуйте чуть позже.")
        raise e


def end_conversation(update, context):
    chat_id = update.message.chat_id

    response = requests.post(f'{api_uri}/end_conversation',
                             json={"user_id": chat_id})

    try:
        response = response.json()
        send_response(update, context, response)
    except Exception as e:
        update.message.reply_text("Простите, мы не можем обработать ваш запрос. Попробуйте чуть позже.")
        raise e


def help(update, context):
    message = f"Бот ParlAI.\n"
    message += f"/start — начать диалог\n"
    message += f"/context <message> — отправить контекстное сообщение (от лица бота)\n"
    message += f"/done — закончить диалог\n"
    message += "Все остальные сообщения будут переданы прямо боту.\n"

    update.message.reply_text(message)


def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    text_handler = MessageHandler(Filters.text, send_message)

    dp.add_handler(CommandHandler("start", start_conversation))
    dp.add_handler(CommandHandler("context", send_person_message, pass_args=True))
    dp.add_handler(CommandHandler("done", end_conversation))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(text_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
