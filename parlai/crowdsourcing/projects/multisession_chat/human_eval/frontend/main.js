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

// import { CustomOnboardingChatApp } from "../../../../../../tasks/model_chat/frontend/components/chat_app_with_onboarding.jsx"
import { CustomOnboardingChatApp } from "./components/chat_app_with_onboarding.jsx"
import { DefaultTaskDescription } from "bootstrap-chat";
// import { ResponseComponent } from "../../../../../../tasks/model_chat/frontend/components/response_panes.jsx";
import { ResponseComponent } from "./components/response_panes.jsx";
import { RenderChatMessage, PersonaLines } from "./components/message.jsx";

function PersonaContext({taskContext, isLeftPane}){
  if (taskContext){
      return (
        <div>
              <span id="image">
                <PersonaLines
              taskData={{personas: taskContext.personas, time_num: taskContext.time_num, time_unit: taskContext.time_unit}}
              isLeftPane={isLeftPane}
              />
              </span>
            </div>
      )
  } else {
    return null;

  }
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
      renderSidePane={({ mephistoContext: { taskConfig }, appContext: { taskContext } }) => (
        <DefaultTaskDescription
          chatTitle={taskConfig.chat_title}
          taskDescriptionHtml={taskConfig.left_pane_text}
        >
              <PersonaContext
              taskContext={taskContext}
              isLeftPane={true}
              />
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

