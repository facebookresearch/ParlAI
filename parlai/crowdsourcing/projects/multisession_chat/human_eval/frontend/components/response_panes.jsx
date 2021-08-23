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
  Col,
  FormGroup,
  ControlLabel,
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
  
function FinalSurvey({ taskConfig, onMessageSend, active, currentCheckboxes}) {
  const [rating, setRating] = React.useState(0);
  const [sending, setSending] = React.useState(false);

  const tryMessageSend = React.useCallback(() => {
    if (active && !sending) {
      setSending(true);
      onMessageSend({ 
        text: "", 
        task_data: {
          problem_data_for_prior_message: currentCheckboxes,
          final_rating: rating,
        },
      }).then(() => {
        setSending(false);
      });
    }
  }, [active, sending, rating, onMessageSend]);

  const ratingOptions = [<option key="empty_option" />].concat(
    ["1", "2", "3", "4", "5"].map((option_label, index) => {
      return (
        <option key={"option_" + index.toString()}>{option_label}</option>
      );
    })
  );

  const ratingSelector = (
    <FormGroup key={"final_survey"}>
      <Col
        componentClass={ControlLabel}
        sm={6}
        style={{ fontSize: "16px" }}
      >
        {taskConfig.final_rating_question}
      </Col>
      <Col sm={5}>
        <FormControl
          componentClass="select"
          style={{ fontSize: "16px" }}
          value={rating}
          onChange={(e) => {
            var val = e.target.value;
            setRating(val);
          }}
          disabled={!active || sending}
        >
          {ratingOptions}
        </FormControl>
      </Col>
    </FormGroup>
  );

  return (
    <div className="response-type-module">
      <div>
        You've completed the conversation. Please annotate the final turn, fill out
        the following, and hit Done.
      </div>
      <br />
      <div className="response-bar">
        {ratingSelector}
        <Button
          className="btn btn-submit submit-response"
          id="id_send_msg_button"
          disabled={rating === 0 || !active || sending}
          onClick={() => tryMessageSend()}
        >
          Done
        </Button>
      </div>
    </div>
  );
}

function CheckboxTextResponse({ onMessageSend, active, currentCheckboxes}) {
  const [textValue, setTextValue] = React.useState("");
  const [sending, setSending] = React.useState(false);

  const inputRef = React.useRef();

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
        task_data: {problem_data_for_prior_message: currentCheckboxes} 
      }).then(() => {
        setTextValue("");
        setSending(false);
      });
    }
  }, [textValue, active, sending, onMessageSend]);

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

  return (
    <div className="response-type-module">
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
  );
}

function ResponseComponent({ taskConfig, appSettings, onMessageSend, active }) {
  
  const lastMessageIdx = appSettings.numMessages - 1;
  const lastMessageAnnotations = appSettings.checkboxValues[lastMessageIdx];
  
  const computedActive = (
    taskConfig.annotation_buckets === null || 
    hasAnyAnnotations(lastMessageAnnotations) & active
  );
  
  if (lastMessageIdx >= taskConfig.min_num_turns * 2) {
    return (
      <FinalSurvey
        onMessageSend={onMessageSend}
        taskConfig={taskConfig}
        active={computedActive}
        currentCheckboxes={lastMessageAnnotations}
      />
    );
  } else {
    return (
      <CheckboxTextResponse 
        onMessageSend={onMessageSend}
        active={computedActive}
        currentCheckboxes={lastMessageAnnotations}
      />
    );
  }
}

export { ResponseComponent };