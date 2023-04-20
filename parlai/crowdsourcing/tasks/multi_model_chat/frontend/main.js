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

import { CustomOnboardingChatApp } from "./components/chat_app_with_onboarding.jsx";
import { TaskDescription } from "./components/sidepane.jsx";
import { ResponseComponent } from "./components/response_panes.jsx";
import { RenderChatMessage } from "./components/message.jsx";

function MainApp() {
  const [needRating, setNeedRating] = React.useState(false);

  function newMessageHandler(messages) {
    const lastMessage = messages.at(messages.length - 1);
    setNeedRating(lastMessage?.needs_rating === true ? true : false);
  }

  function TextResponse({ taskConfig, appSettings, onMessageSend, active }) {
    return (
      <ResponseComponent
        appSettings={appSettings}
        taskConfig={taskConfig}
        activeText={active}
        activeRating={needRating}
        onMessageSend={onMessageSend}
      />
    );
  }

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
      /* eslint-disable no-unused-vars */
      renderSidePane={({
        mephistoContext: { taskConfig },
        appContext: { taskContext }
      }) => <TaskDescription context={taskContext} />}
      /* eslint-enable no-unused-vars */
      renderTextResponse={({
        mephistoContext: { taskConfig },
        appContext: { appSettings },
        onMessageSend,
        active
      }) => (
        <TextResponse
          appSettings={appSettings}
          taskConfig={taskConfig}
          active={active}
          onMessageSend={onMessageSend}
        />
      )}
      onMessagesChange={messages => newMessageHandler(messages)}
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
