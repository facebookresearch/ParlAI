/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import "bootstrap-chat/styles.css";

import { CustomOnboardingChatApp } from "./components/chat_app_with_onboarding.jsx"
import { DefaultTaskDescription } from "bootstrap-chat";
import { Checkboxes } from './components/checkboxes.jsx';

import { 
  FormControl, 
  Button,
  Col,
  FormGroup,
  ControlLabel,
} from "react-bootstrap";

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
  }, [active, sending, onMessageSend]);

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

function ReponseComponent({ taskConfig, onMessageSend, active, currentCheckboxes, lastMessageIdx}) {
  if (lastMessageIdx >= taskConfig.min_num_turns * 2) {
    return (
      <FinalSurvey
        onMessageSend={onMessageSend}
        taskConfig={taskConfig}
        active={active}
        currentCheckboxes={currentCheckboxes}
      />
    );
  } else {
    return (
      <CheckboxTextResponse 
        onMessageSend={onMessageSend}
        active={active}
        currentCheckboxes={currentCheckboxes}
      />
    );
  }
}

function MaybeCheckboxChatMessage({ isSelf, duration, agentName, message = "", checkbox = null }) {
  const floatToSide = isSelf ? "right" : "left";
  const alertStyle = isSelf ? "alert-info" : "alert-warning";

  return (
    <div className="row" style={{ marginLeft: "0", marginRight: "0" }}>
      <div
        className={"alert message " + alertStyle}
        role="alert"
        style={{ float: floatToSide }}
      >
        <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }}>
          <b>{agentName}</b>: <span dangerouslySetInnerHTML={{ __html: message }}></span>
        </span>
        {checkbox}
      </div>
    </div>
  );
}

function RenderChatMessage({ message, mephistoContext, appContext, idx }) {
  const { agentId, taskConfig } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;
  const { appSettings, setAppSettings } = appContext;
  const { checkboxValues } = appSettings;
  const isHuman = (message.id === agentId || message.id == currentAgentNames[agentId]);
  const annotationBuckets = taskConfig.annotation_buckets;
  const annotationIntro = taskConfig.annotation_question;

  var checkboxes = null;
  if (!isHuman) {
    let thisBoxAnnotations = checkboxValues[idx];
    if (!thisBoxAnnotations) {
      thisBoxAnnotations = Object.fromEntries(
        annotationBuckets.map(bucket => [bucket.value, false])
      )
    }
    checkboxes = <div style={{"fontStyle": "italic"}}>
      <br />
      {annotationIntro}
      <br />
      <Checkboxes 
        annotations={thisBoxAnnotations} 
        onUpdateAnnotations={
          (newAnnotations) => {
            checkboxValues[idx] = newAnnotations;
            setAppSettings({checkboxValues});
          }
        } 
        annotationBuckets={annotationBuckets} 
        turnIdx={idx} 
        askReason={false} 
        enabled={idx == appSettings.numMessages - 1}
      />
    </div>;
  }
  return (
    <MaybeCheckboxChatMessage
      isSelf={isHuman}
      agentName={
        message.id in currentAgentNames
          ? currentAgentNames[message.id]
          : message.id
      }
      message={message.text}
      taskData={message.task_data}
      messageId={message.message_id}
      checkbox={checkboxes}
    />
  );
}

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

function MainApp() {
  return (
    <CustomOnboardingChatApp
      propAppSettings={{ checkboxValues: {} }}
      renderMessage={({ message, idx, mephistoContext, appContext }) => (
        <RenderChatMessage
          message={message}
          mephistoContext={mephistoContext}
          appContext={appContext}
          idx={idx}
          key={message.message_id + "-" + idx}
        />
      )}
      renderSidePane={({ mephistoContext: { taskConfig } }) => (
        <DefaultTaskDescription
          chatTitle={taskConfig.chat_title}
          taskDescriptionHtml={taskConfig.task_description}
        />
      )}
      renderTextResponse={
        ({ 
          mephistoContext: { taskConfig }, 
          appContext: { appSettings },
          onMessageSend,
          active,

        }) => {
          const currLastMessageIdx = appSettings.numMessages - 1;
          const lastMessageAnnotations = appSettings.checkboxValues[currLastMessageIdx];
          
          return (
            <ReponseComponent 
              lastMessageIdx={appSettings.numMessages - 1}
              taskConfig={taskConfig}
              currentCheckboxes={lastMessageAnnotations}
              active={hasAnyAnnotations(lastMessageAnnotations) & active}
              onMessageSend={onMessageSend}
            />
          )
        }    
      }
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));

