import requests

initial_message = 'hello'

requests.post('http://84.201.158.229:8080/api/start_conversation',
              json={'user_id': 1})
requests.post('http://84.201.158.229:8080/api/start_conversation',
              json={'user_id': 2})
requests.post('http://84.201.158.229:8080/api/start_conversation',
              json={'user_id': 1})
requests.post('http://84.201.158.229:8080/api/start_conversation',
              json={'user_id': 2})

r = requests.post('http://84.201.158.229:8080/api/send_message',
              json={'user_id': 1, 'message_text': initial_message})
response = r.json()
print('Bot 2:', initial_message)

while True:
    text = response['text']
    print('Bot 1:', text)

    r = requests.post('http://84.201.158.229:8080/api/send_message',
              json={'user_id': 2, 'message_text': text})

    response = r.json()
    text = response['text']
    print('Bot 2:', text)

    r = requests.post('http://84.201.158.229:8080/api/send_message',
              json={'user_id': 2, 'message_text': text})
    response = r.json()
    
    
