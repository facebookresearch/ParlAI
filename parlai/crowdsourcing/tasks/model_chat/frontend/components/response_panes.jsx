/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

import InputEmoji from 'react-input-emoji'
import { Button, Col, ControlLabel, Form, FormControl, FormGroup } from "react-bootstrap";


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

function RatingSelector({ active, ratings, sending, ratingQuestion, ratingIndex, setRatings }) {
  const ratingOptions = [<option key="empty_option" />].concat(
    ["1", "2", "3", "4", "5"].map((option_label, index) => {
      return (
        <option key={"option_" + index.toString()}>{option_label}</option>
      );
    })
  );

  function handleRatingSelection(val) {
    const newRatings = ratings.map((item, index) => {
      if (index === ratingIndex) {
        return val;
      } else {
        return item;
      }
    });
    setRatings(newRatings);
  }

  return (
    <FormGroup key={"final_survey_" + ratingIndex.toString()}>
      <Col
        componentClass={ControlLabel}
        sm={6}
        style={{ fontSize: "14px" }}
      >
        {ratingQuestion}
      </Col>
      <Col sm={5}>
        <FormControl
          componentClass="select"
          style={{ fontSize: "14px" }}
          value={ratings[ratingIndex]}
          onChange={(e) => handleRatingSelection(e.target.value)}
          disabled={!active || sending}
        >
          {ratingOptions}
        </FormControl>
      </Col>
    </FormGroup>
  );
}

function FinalSurvey({ taskConfig, onMessageSend, active, currentCheckboxes }) {
  const [sending, setSending] = React.useState(false);

  // Set up multiple response questions
  let ratingQuestions = taskConfig.final_rating_question.split("|");
  let initialRatings = [];
  for (let _ of ratingQuestions) {
    initialRatings.push("");
  }
  const [ratings, setRatings] = React.useState(initialRatings)

  const tryMessageSend = React.useCallback(() => {

    let all_ratings_filled = ratings.every((r) => r !== "");
    let rating = ratings.join('|');

    if (all_ratings_filled && active && !sending) {
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
  }, [active, sending, ratings, onMessageSend]);

  const listRatingSelectors = ratingQuestions.map((ratingQuestion, ratingIndex) => {
    return (
      <RatingSelector
        active={active}
        ratings={ratings}
        sending={sending}
        ratingQuestion={ratingQuestion}
        ratingIndex={ratingIndex}
        setRatings={setRatings}
      >
      </RatingSelector>
    );
  });

  if (listRatingSelectors.length > 1) {
    // Show ratings to the right of the questions
    return (
      <div className="response-type-module">
        <div>
          You've completed the conversation. Please annotate the final turn, fill out
          the following, and hit Done.
        </div>
        <br />
        <Form
          horizontal
        >
          {listRatingSelectors}
          <Button
            className="btn btn-submit submit-response"
            id="id_send_msg_button"
            disabled={!active || sending}
            onClick={() => tryMessageSend()}
          >
            Done
          </Button>
        </Form>
      </div>
    );
  } else {
    // Show the single rating below the single question
    return (
      <div className="response-type-module">
        <div>
          You've completed the conversation. Please annotate the final turn, fill out
          the following, and hit Done.
        </div>
        <br />
        <div className="response-bar">
          {listRatingSelectors}
          <Button
            className="btn btn-submit submit-response"
            id="id_send_msg_button"
            disabled={!active || sending}
            onClick={() => tryMessageSend()}
          >
            Done
          </Button>
        </div>
      </div>
    );
  }
}

function CheckboxTextResponse({ onMessageSend, taskConfig, active, currentCheckboxes }) {
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
        task_data: { problem_data_for_prior_message: currentCheckboxes }
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

  if (taskConfig.emoji_picker){
    return (
      <div className="response-type-module">
        <div className="response-bar">
        <InputEmoji
            value={textValue}
            className="response-text-input"
            onEnter={(e) => handleKeyPress(e)}
            onChange={setTextValue}
            placeholder="Please enter here..."
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
  } else {
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
        taskConfig={taskConfig}
        active={computedActive}
        currentCheckboxes={lastMessageAnnotations}
      />
    );
  }
}

export { ResponseComponent };