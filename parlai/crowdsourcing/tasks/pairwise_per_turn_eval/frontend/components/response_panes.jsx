/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

import { 
  FormControl, 
  Button,
  FormGroup,
  Radio
} from "react-bootstrap";

function hasAnyAnnotations(annotations) {
  if (!annotations) {
    return false;
  }
  for (const key in annotations) {
    if (annotations[key] === true) {
      return true;
    }
  }
  return false;
}

/*
This function makes it so that when the two bots' messages are receive, users are first
able to choose between the messages and provide a justification value for that choice,
and then type their own response afterwards. Specifically, on task_turn_idx % 2 == 0,
the response panel becomes bot choice + justification and on task_turn_idx % 2 == 1, the
response panel becomes human response.

All of the data pertaining to the user are received by the frontend using taskConfig,
and sent to the backend using the various tryXXXXSend callbacks.

*/

function CheckboxTextResponse({ taskConfig, taskContext, onMessageSend, active, currentCheckboxes}) {
  const [expectedTurnValue, setExpectedTurnValue] = React.useState(0);
  const [chosenBotResponseValue, setChosenBotResponseValue] = React.useState({});
  const [chosenBotIdValue, setChosenBotIdValue] = React.useState("");
  const [justificationValue, setJustificationValue] = React.useState("");
  const [textValue, setTextValue] = React.useState("");
  const [sending, setSending] = React.useState(false);

  const inputRef = React.useRef();

  const top_bot_data = taskContext.top_bot_data;
  const bottom_bot_data = taskContext.bottom_bot_data;

  const top_bot_id = top_bot_data === undefined ? "" : top_bot_data.top_bot_id
  const bottom_bot_id = bottom_bot_data === undefined ? "" : bottom_bot_data.bottom_bot_id
  const top_bot_response = top_bot_data === undefined ? null : top_bot_data.top_bot_response
  const bottom_bot_response = bottom_bot_data === undefined ? null : bottom_bot_data.bottom_bot_response
  const top_bot_response_text = top_bot_data === undefined ? "" : top_bot_data.top_bot_response.text
  const bottom_bot_response_text = bottom_bot_data === undefined ? "" : bottom_bot_data.bottom_bot_response.text

  const task_turn_idx = taskContext.task_turn_idx === undefined ? -1 : taskContext.task_turn_idx

  React.useEffect(() => {
    if (active && inputRef.current && inputRef.current.focus) {
      inputRef.current.focus();
    }
  }, [active]); 

  const tryMessageSend = React.useCallback(() => {
    if (textValue !== "" && active && !sending) {
      setSending(true);
      onMessageSend({ 
        text: textValue, 
        task_data: {
          problem_data_for_prior_message: currentCheckboxes,
          'task_turn_idx': task_turn_idx,
          'text_value': textValue
        } 
      }).then(() => {
        setTextValue("");
        setSending(false);
        setExpectedTurnValue(expectedTurnValue + 1);
      });
    }
  }, [textValue, active, sending, onMessageSend]);

  const tryFinalJustificationSend = React.useCallback(() => {
    if (justificationValue !== "" && active && !sending) {
      setSending(true);
      onMessageSend({
        text: '', 
        task_data: {
          problem_data_for_prior_message: currentCheckboxes,
          'accepted_bot_response': chosenBotResponseValue,
          'not_accepted_bot_response': chosenBotResponseValue === top_bot_response ? bottom_bot_response : top_bot_response,
          'accepted_bot_id': chosenBotIdValue,
          'not_accepted_bot_id': chosenBotIdValue === top_bot_id ? bottom_bot_id : top_bot_id,
          'task_turn_idx': task_turn_idx,
          'justification_value': justificationValue,
          'finished': true
        },
      }).then(() => {
        setTextValue("");
        setSending(false);
      });
    }
  }, [justificationValue, active, sending, onMessageSend]);

  const tryJustificationSend = React.useCallback(() => {
    console.log(chosenBotResponseValue)
    if (Object.keys(chosenBotResponseValue).length == 0) {
      alert("Please choose a response and write your justification before proceeding!");
      setJustificationValue("");
    } else if (justificationValue !== "" && active && !sending) {
      console.log("Trying to send message!")
      setSending(true);
      onMessageSend({ 
        text: '', 
        task_data: {
          problem_data_for_prior_message: currentCheckboxes,
          'accepted_bot_response': chosenBotResponseValue,
          'not_accepted_bot_response': chosenBotResponseValue === top_bot_response ? bottom_bot_response : top_bot_response,
          'accepted_bot_id': chosenBotIdValue,
          'not_accepted_bot_id': chosenBotIdValue === top_bot_id ? bottom_bot_id : top_bot_id,
          'task_turn_idx': task_turn_idx,
          'justification_value': justificationValue
        } 
      }).then(() => {
        setJustificationValue("");
        setChosenBotResponseValue({});
        setChosenBotIdValue("");
        setSending(false);
        setExpectedTurnValue(expectedTurnValue + 1);
      });
    }
  }, [justificationValue, active, sending, onMessageSend]);

  const handleKeyPress = React.useCallback(
    (e) => {
      if (e.key === "Enter") {
        tryMessageSend();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
      }
    },
    [tryMessageSend]
  );

  const handleFinalJustificationKeyPress = React.useCallback(
    (e) => {
      if (e.key === "Enter") {
        tryFinalJustificationSend();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
      }
    },
    [tryFinalJustificationSend]
  );

  const handleJustificationKeyPress = React.useCallback(
    (e) => {
      if (e.key === "Enter") {
        tryJustificationSend();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
      }
    },
    [tryJustificationSend]
  );

  const handleRadioChange = e => {
    e.persist();
    if (e.target.value === top_bot_response_text) {
      setChosenBotResponseValue(top_bot_response);
      setChosenBotIdValue(top_bot_id);
    } else if (e.target.value === bottom_bot_response_text) {
      setChosenBotResponseValue(bottom_bot_response);
      setChosenBotIdValue(bottom_bot_id);
    } else {
      alert("Something went extremely wrong. Please contact the requestors reporting this error.")
    }
  };

  if (task_turn_idx !== -1 && task_turn_idx >= 2 * (taskConfig.min_num_turns - 1) ) {
    /*
    We want to ask the human to choose between bot responses min_num_turns times. For
    each of those times, task_turn_idx will be equal to 0, 2, ..., 2*(min_num_turns-1).
    */
    return (
      <div className="response-type-module">
          <div>
            <b>{taskContext["prompt_instruction"]}</b>
          </div>
          
          <FormGroup>
            <Radio
              name="top_bot_response"
              value={top_bot_response_text}
              onChange={handleRadioChange}
              checked={chosenBotResponseValue === top_bot_response}
              >
                {top_bot_response_text}
            </Radio>
            <Radio
              name="bottom_bot_response"
              value={bottom_bot_response_text}
              onChange={handleRadioChange}
              checked={chosenBotResponseValue === bottom_bot_response}
              >
                {bottom_bot_response_text}
            </Radio>
          </FormGroup>
  
          <div>
            <b>Please provide a brief justification for your choice (a few words or a sentence):</b>
          </div>
  
          <div className="response-bar">
            <FormControl
              type="text"
              className="response-text-input"
              inputRef={(ref) => {
                inputRef.current = ref;
              }}
              value={justificationValue}
              placeholder="Please enter here..."
              onKeyPress={(e) => handleFinalJustificationKeyPress(e)}
              onChange={(e) => setJustificationValue(e.target.value)}
              disabled={!active || sending}
            />
          <Button
            className="btn btn-primary submit-response"
            id="id_send_msg_button"
            disabled={justificationValue === "" || !active || sending}
            onClick={() => tryFinalJustificationSend()}
          >
            Send
          </Button>
        </div>
      </div>
    );
  } else {
    
    if (task_turn_idx !== expectedTurnValue) {
      return (
      <div>
      </div>
      );
    } else if (task_turn_idx % 2 === 0) {
      return (
        <div className="response-type-module">
          <div>
            <b>{taskContext["prompt_instruction"]}</b>
          </div>
          
          <FormGroup>
            <Radio
              name="top_bot_response"
              value={top_bot_response_text}
              onChange={handleRadioChange}
              checked={chosenBotResponseValue === top_bot_response}
              >
                {top_bot_response_text}
            </Radio>
            <Radio
              name="bottom_bot_response"
              value={bottom_bot_response_text}
              onChange={handleRadioChange}
              checked={chosenBotResponseValue === bottom_bot_response}
              >
                {bottom_bot_response_text}
            </Radio>
          </FormGroup>
  
          <div>
            <b>Please provide a brief justification for your choice (a few words or a sentence):</b>
          </div>
  
          <div className="response-bar">
            <FormControl
              type="text"
              className="response-text-input"
              inputRef={(ref) => {
                inputRef.current = ref;
              }}
              value={justificationValue}
              placeholder="Please enter here..."
              onKeyPress={(e) => handleJustificationKeyPress(e)}
              onChange={(e) => setJustificationValue(e.target.value)}
              disabled={!active || sending}
            />
            <Button
              className="btn btn-primary submit-response"
              id="id_send_msg_button"
              disabled={justificationValue === "" || !active || sending}
              onClick={() => tryJustificationSend()}
            >
              Send
            </Button>
          </div>
        </div>
      );
    } else {
      return (
        <div className="response-type-module">
          <div>
            <b>Please enter a follow-up response to your partner:</b>
          </div>
          <div className="response-bar">
            <FormControl
              type="text"
              className="response-text-input"
              inputRef={(ref) => {
                inputRef.current = ref;
              }}
              value={textValue}
              placeholder="Please enter here..."
              onKeyPress={(e) => handleKeyPress(e)}
              onChange={(e) => setTextValue(e.target.value)}
              disabled={!active || sending}
            />
            <Button
              className="btn btn-primary submit-response"
              id="id_send_msg_button"
              disabled={textValue === "" || !active || sending}
              onClick={() => tryMessageSend()}
            >
              Send
            </Button>
          </div>
        </div>
      )
    }
  }


}

function ResponseComponent({ taskConfig, appSettings, onMessageSend, active, appContext }) {
  const lastMessageIdx = appSettings.numMessages - 1;
  const lastMessageAnnotations = appSettings.checkboxValues[lastMessageIdx];
  
  const computedActive = (
    taskConfig.annotation_buckets === null || 
    hasAnyAnnotations(lastMessageAnnotations) & active
  );
  /*
  TODO: why is computedActive no longer being used?
   */

  return (
    <CheckboxTextResponse
      taskConfig={taskConfig}
      taskContext={appContext.taskContext}
      onMessageSend={onMessageSend}
      active={true}
      currentCheckboxes={lastMessageAnnotations}
      expectedTurn={0}
    />
  );
}

export { ResponseComponent };