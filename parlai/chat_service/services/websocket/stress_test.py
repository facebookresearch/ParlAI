import threading
import requests
import random
from time import time, sleep


times = []
#print(requests.post('http://84.201.158.229:8080/api/start_conversation').json())

def print_times():
    while True:
        times_len = len(times)
        print(times_len, end=' ')
        if times_len == THREAD_COUNT * REQUESTS_COUNT:
            print()
            print(sum(times) / len(times))
            return

        sleep(15)
    
THREAD_COUNT = 1
REQUESTS_COUNT = 3

def task():
    for i in range(REQUESTS_COUNT):
        start_time = time()

        response = requests.post('http://84.201.158.229:8080/api/send_message',
                                 json={'message_text': 'what is your favourite fruit?',
                                       'message_history': ["i love apples", "your persona: i love airplanes", "hello", "how's going?", "i love video games", "i love skyrim", "your persona: i hate bananas", "steve jobs is genius", "do you have children?", "i am busy"] * 100})
        print(response.json())
        end_time = time()

        times.append(end_time - start_time)


threads = []
for i in range(THREAD_COUNT):
    threads.append(threading.Thread(target=task))
threads.append(threading.Thread(target=print_times))

for thread in threads:
    thread.start()
        
