from parlai.core.teachers import DialogTeacher
from .build import build
import os, copy



class QASRLTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'qasrl'

        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'QA-SRL')
        self.opt = copy.deepcopy(opt)

        super().__init__(opt, shared)


    def setup_data (self, input_path):

        print('loading: ' + input_path)
        file_path = os.path.join(input_path, 'wiki1.train.qa')

        new_episode = True

        def convert_to_qa(input_data):
            lines = input_data.split('\n')
            context = lines[1]
            predicate_count =  int(lines[0].split('\t')[-1])
            unparsed_qa = lines[2:]
            def parse_qa(qa_line):
                qa_split = qa_line.split('\t?\t')
                question = context + '\n' + qa_split[0].replace('\t_', '').replace('\t',' ')+ '?'
                answers = qa_split[1].split(' ### ')
                return [question, answers]

            qa_pairs = []
            counter = 0
            for i in range(predicate_count):
                question_count = int(unparsed_qa[counter].split('\t')[-1])
                counter += 1
                for j in range(question_count):
                    qa_pairs.append(parse_qa(unparsed_qa[counter]))
                    counter += 1
            return qa_pairs



        with open(file_path) as file:
            #split the data by sentences
            file_data = file.read().split('\n\n')[:-1]
        for data in file_data:
            for qa in convert_to_qa(data):
                question = qa[0]
                answers = qa[1]
                yield (question, answers, None, None), new_episode



class DefaultTeacher(QASRLTeacher):
    pass

