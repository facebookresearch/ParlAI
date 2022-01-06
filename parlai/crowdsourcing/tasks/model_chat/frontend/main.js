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
import { ResponseComponent } from "./components/response_panes.jsx";
import { RenderChatMessage } from "./components/message.jsx";

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
      renderSidePane={({ mephistoContext: { taskConfig }, appContext: { taskContext } }) => (
        <DefaultTaskDescription
          chatTitle={taskConfig.task_title}
          taskDescriptionHtml={
              taskConfig.left_pane_text.replace(
                  "[persona_string_1]", taskContext.human_persona_string_1,
              ).replace(
                  "[persona_string_2]", taskContext.human_persona_string_2,
              )
          }
        >
          {(taskContext.hasOwnProperty('image_src') && taskContext['image_src']) ? (
            <div>
              <h4>Conversation image:</h4>
              <span id="image">
                <img src={taskContext.image_src} alt='Image'/>
              </span>
              <br />
            </div>
          ) : null}
        </DefaultTaskDescription>
      )}
      renderTextResponse={
        ({
          mephistoContext: { taskConfig },
          appContext: { appSettings },
          onMessageSend,
          active,

        }) => (
          <ResponseComponent
            appSettings={appSettings}
            taskConfig={taskConfig}
            active={active}
            onMessageSend={onMessageSend}
          />
        )
      }
    />
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
