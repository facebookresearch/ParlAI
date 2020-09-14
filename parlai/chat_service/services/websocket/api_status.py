import requests
import time
from telegram.ext import Updater

TOKEN = '1399687512:AAHFOYX2FM2ei1pmsbr0QdFcu3dVtumRdXQ'
API_URI = "http://84.201.158.229:8080/api/send_message"
IMAGE_LINK = "https://sun9-8.userapi.com/ByUCwAmCChHMQgDlUDrda8T1snCstfXL3vJNRg/rRtuRR9AM_M.jpg"


def sendImage():
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto";
    files = {'photo': open('kit.jpg', 'rb')}
    data = {'chat_id' : "YOUR_CHAT_ID"}
    r= requests.post(url, files=files, data=data)
    print(r.status_code, r.reason, r.content)


updater = Updater(TOKEN)

while True:
    try:
        response = requests.get(API_URI,
                                json={"message_text": "what is your favourite fruit?", "message_history":
                                      ["your persona: my name is Karen", "your persona: i am 20 years girl",
                                       "your persona: my job is IT Manager", "your persona: my hobby is Lace making",
                                       "your persona: my hobby is Fly tying", " hello", "Hello, how's going?",
                                       " I'm fine, thanks", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                                       "12", "13", "14", "15", "16", "17", "18", "19", "20"]},
                                timeout=60)
    except Exception as e:
        updater.bot.send_photo(chat_id=-428722445, photo=IMAGE_LINK)

    time.sleep(600)



