Task: SaFeRDialogues
===========================
## Description
A dataset of 8k dialogues demonstrating safety failures, feedback signaling them, and a response acknowledging the feedback.

Dataset has been released under the CC BY-NC license. Please refer to the LICENSE file in this folder for more information.

[ArXiv Paper](https://arxiv.org/abs/2110.07518)

## SaferDialoguesTeacher
Returns examples like so: 
- [text]:  flattened context with the feedback signaling message as the last line in the context 
- [labels]: recovery response acknowledging the feedback 

Note: The dataset is flattened, so there is one example per episode. 

If the `--recovery` flag is set to `false` (`true` by default) then the recovery response is omitted and the labels contains the signaling message and the text contains the context lines before that.

Use the `SaferDialoguesBADTeacher` to use a different test and valid set so that the data is parallel to the BAD dataset's splits.

Tags: #SaFeRDialogues, #All, #Recovery, #Safety, #ChitChat
