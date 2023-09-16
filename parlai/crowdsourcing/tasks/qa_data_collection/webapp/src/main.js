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
import ResizableTextArea from "react-fluid-textarea";

import { ChatApp, ChatMessage, DefaultTaskDescription } from "bootstrap-chat";

function RenderChatMessage({ message, mephistoContext, appContext }) {
  const { agentId } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;

  return (
    <ChatMessage
      isSelf={message.id === agentId || message.id in currentAgentNames}
      agentName={
        message.id in currentAgentNames
          ? currentAgentNames[message.id]
          : message.id
      }
      message={message.text}
      taskData={message.task_data}
      messageId={message.message_id}
    />
  );
}

function logSelection(event) {
  const selection = event.target.value.substring(
    event.target.selectionStart,
    event.target.selectionEnd
  );
  console.log(selection);
}

function Passage({ passage }) {
  // Formatting to make textarea look like div, span selection works best on textarea
  const mystyle = {
    outline: "none",
    backgroundColor: "#dff0d8",
    width: "100%",
    border: "0px"
  };
  if (passage) {
    return (
      <div>
        <h2>Passage</h2>
        <ResizableTextArea
          defaultValue={passage}
          readOnly
          style={mystyle}
          onClick={logSelection}
        />
      </div>
    );
  }
  return null;
}

function MainApp() {
  const [passage, setPassage] = React.useState("");

  // Currently no way to display task description without changing Mephisto files
  return (
    <ChatApp
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
        >
          <Passage passage={passage} />
        </DefaultTaskDescription>
      )}
      onMessagesChange={messages => {
        if (messages.length > 0 && "passage" in messages[messages.length - 1]) {
          console.log("setting passage");
          setPassage(messages[messages.length - 1].passage);
        }
      }}
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
