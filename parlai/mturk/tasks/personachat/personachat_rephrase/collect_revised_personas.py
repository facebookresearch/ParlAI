import os
import pickle

done = [False for _ in range(102)]

root = '/private/home/edinan/ParlAI/data/personachat_rephrase/rephrased_personas/'
for filename in os.listdir(root):
    if 'incomplete' not in str(filename) and 'sandbox' not in str(filename):
        print(root + str(filename))
        p_dict = pickle.load(open(root + str(filename), 'rb'))
        print("PERSONA IDX: ", p_dict['persona_idx'])
        done[int(p_dict['persona_idx'])] = True
        print("WORKER: ", p_dict['worker_id'], '\n')
        print("ORIGINAL PERSONA:")
        for pers in p_dict['persona']:
            print(pers)
        print("\n")
        print("REPHRASED PERSONA:")
        for pers in p_dict['rephrased']:
            print(pers)
        print("---------------------------------\n")

for idx, fin in enumerate(done):
    if not fin:
        print(idx)
