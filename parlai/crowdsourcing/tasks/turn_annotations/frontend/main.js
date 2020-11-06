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
import { ChatMessage, DefaultTaskDescription } from "bootstrap-chat";
import { Checkboxes } from './components/checkboxes.jsx';

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
  const isHuman = (message.id === agentId || message.id == currentAgentNames[agentId])
  const annotationBuckets = taskConfig.annotation_buckets;
  const annotationIntro = taskConfig.annotation_question;
  const [currentTurnAnnotations, setCurrentAnnotations] = React.useState(
    Object.fromEntries(
      annotationBuckets.map(bucket => [bucket.value, false])
    )
  );

  var checkboxes = null;
  if (!isHuman) {
    checkboxes = <div style={{"fontStyle": "italic"}}>
      <br />
      {annotationIntro}
      <br />
      <Checkboxes 
        annotations={currentTurnAnnotations} 
        onUpdateAnnotations={setCurrentAnnotations} 
        annotationBuckets={annotationBuckets} 
        turnIdx={idx} 
        askReason={false} 
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

function MainApp() {
  return (
    <CustomOnboardingChatApp
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
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
