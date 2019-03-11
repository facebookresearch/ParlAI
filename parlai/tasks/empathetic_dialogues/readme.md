Dataset has been released under the CC BY-NC license.

EmpatheticDialogueTeacher returns examples like so:
- [text]:  context line (previous utterance by 'speaker')
- [labels]: label line  (current utterance by 'listener')
with additional task specific fields:
- [situation]: a 1-3 sentence description of the situation that the conversation is 
- [emotion]: one of 32 emotion words

Other optional fields:
- [prepend_ctx]: fasttext prediction on context line - or None
- [prepend_cand]: fasttext prediction on label line (candidate) - or None
- [deepmoji_ctx]: vector encoding from deepmoji penultimate layer - or None
- [deepmoji_cand]: vector encoding from deepmoji penultimate layer for label line (candidate) - or None
