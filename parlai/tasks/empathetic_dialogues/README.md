Task: Empathetic Dialogues
===========================
Description: A dataset of 25k conversations grounded in emotional situations to facilitate training and evaluating dialogue systems. See https://arxiv.org/abs/1811.00207 for more information. 
=========================== 

LICENSE: This dataset has been released under the CC-BY-4.0 License. Please
refer to the LICENSE_DOCUMENTATION file in this repository for more
information.

## EmpatheticDialoguesTeacher
Returns examples like so: 
- [text]:  context line (previous utterance by 'speaker') 
- [labels]: label line  (current utterance by 'listener') 
with additional task specific fields: 
- [situation]: a 1-3 sentence description of the situation that the conversation is 
- [emotion]: one of 32 emotion words 

Tags: #EmpatheticDialogues, #All, #ChitChat
