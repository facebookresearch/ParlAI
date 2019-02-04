EmpatheticDialogueTeacher returns examples like so:
- [emotion]: one of 32 emotion words
- [situation]: a 1-3 sentence description of the situation that the conversation is 
- [text]:  context line (previous utterance by 'speaker')
- [labels]: label line  (current utterance by 'listener')

Optional outputs:
- [prepend_ctx]: fasttext prediction on context line - or None
- [prepend_cand]: fasttext prediction on label line (candidate) - or None
- [deepmoji_ctx]: vector encoding from deepmoji penultimate layer - or None
- [deepmoji_cand]: vector encoding from deepmoji penultimate layer for label line (candidate) - or None

# Adding optional outputs

## fastText predictions (for prepending predicted style labels)

Add 'prepend' and 'fasttextloc' to opt (command line args):
- 'prepend': should be integer (number of top-n labels to return from a fasttext model)
- 'fasttextloc': should be string (location of a pretrained fasttext model)

## Deepmoji encodings
Add 'deepmoji' to opt (command line args):
- 'deepmoji': should be string (location of encodings from some model - we used deepmoji)
