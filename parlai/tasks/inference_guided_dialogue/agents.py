from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from typing import Optional

from. build import build 
from loguru import logger 
from pathlib import Path 
import random 
import json

def split_qa(qa_text): 

    if "?" not in qa_text: 
        logger.error(f"`?` not found in question and answer pair")  
        return None 

    qm_index = qa_text.index("?")

    question = qa_text[:qm_index + 1]
    answer = qa_text[qm_index+1:]

    return (question, answer)

class InferenceGuidedDialogueTeacher(DialogTeacher): 

    def __init__(self, opt, shared=None): 
        self.id = "inf_dial"
        self.datatype = opt['datatype']
        dpath = build(opt)
        opt['datafile'] = Path(dpath) /  "batch_1_prompt_valid.json"
        self.generation_target = opt.get("generation_target")

        super().__init__(opt, shared)


    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Inference-Guided Dialogue Arguments')
        agent.add_argument(
            '-gt',
            '--generation_target',
            type=str,
            default="response",
            choices=['response', 'infq_aresponse', 'infqa_response'],
            help='Targets to use for generation. Refer to README for more information: parlai/tasks/inference_guided_dialogue/README.md',
        )

        return parser 

    def setup_data(self, path): 

        """
        Example: 
        <speaker1> Cameron knew I was dishonest! \n
        <speaker2> You should have behaved better! \n
        <speaker1> Hmm. Cameron now know when I lied to him. \n
        <infq> How to describe <speaker1>? <infa> Unable to be trusted, deceitful \n 
        <speaker2> I cant believe you lied to your friend.
        """

        print(f"Loading: {path}")
        with open(path, "r") as f: 
            self.data = json.load(f) 

        # if self.datatype == "train": 
        # random.shuffle(self.data)
        
        processed_data = [] 
        for d in self.data: 
            dial_hist = d['utterance']
            question, answer = split_qa(d["triple_NL"])
            response = d['response']

            processed = [] 
            for idx, turn in enumerate(dial_hist): 
                speaker_label = "<speaker2>" if idx % 2 else "<speaker1>"
                processed.append(f"{speaker_label} {turn}")

            processed_inf_q = f"<infq> {question}"
            processed_inf_a = f"<infa> {answer}"

            # add question generation 
            if self.generation_target == "infqa_response": 
                input_text = '\n'.join(processed)
                output_text = [processed_inf_q]
                new_episode = True 
                processed_data.append((input_text, output_text, new_episode))

            # add question answer generation 
            if self.generation_target in ["infq_aresponse", "infqa_response"]: 
                input_text = '\n'.join(processed + [processed_inf_q])
                output_text = [processed_inf_a] # provide as list of candidates
                new_episode = True 

                processed_data.append((input_text, output_text, new_episode))

            # default: response generation only 
            input_text = '\n'.join(processed + [processed_inf_q, processed_inf_a])
            output_text = [response] # provide as list of candidates
            new_episode = True 

            processed_data.append((input_text, output_text, new_episode))


        for it, ot, new_ep in processed_data: 
            yield {
                "text": it,
                "labels": ot,
            }, new_ep


            



                    
                
            




class DefaultTeacher(InferenceGuidedDialogueTeacher): 
    pass
