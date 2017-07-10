# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""
You should run this test in clusters where there is less limitation on the number of outbound requests per second.
"""
import requests
import json
import time
import sys
from joblib import Parallel, delayed

num_concurrent_requests = int(sys.argv[1])
recurrent_burst = len(sys.argv) > 2 and sys.argv[2] == '--repeat'
wait_time_between_burst = 1 # in seconds

task_group_id = 'model_evaluator_1499629716'
db_last_message_id = 1000
json_api_endpoint_url = 'https://5lu5k95kwf.execute-api.us-east-1.amazonaws.com/prod/json'


global get_request
def get_request(request_id):
    print("Testing request: "+str(request_id))
    params = {
        'method_name': 'get_new_messages',
        'task_group_id': task_group_id,
        'last_message_id': db_last_message_id,
    }
    response = requests.get(json_api_endpoint_url, params=params, allow_redirects=False)
    try:
        ret = json.loads(response.json())
        print('Success: '+str(ret))
    except Exception as e:
        print(response.content)
        raise e

while True:
    results = Parallel(n_jobs=num_concurrent_requests, backend='threading')(delayed(get_request)(request_id) for request_id in range(num_concurrent_requests))
    print("One run completed")
    if recurrent_burst:
        time.sleep(wait_time_between_burst)
    else:
        break

