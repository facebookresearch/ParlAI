/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Copies code directly from `bootstrap-chat` in order to make a chat app
// that renders with a different frontend onboarding flow.

import React from "react";

import {
  MephistoContext,
  useMephistoLiveTask,
  AGENT_STATUS,
  STATUS_TO_TEXT_MAP,
} from "mephisto-task";
import { BaseFrontend, AppContext } from "bootstrap-chat";
import { OnboardingComponent } from "./onboarding_components.jsx"

/* ================= Application Components ================= */

const INPUT_MODE = {
  WAITING: "waiting",
  INACTIVE: "inactive",
  DONE: "done",
  READY_FOR_INPUT: "ready_for_input",
};

function CustomOnboardingChatApp({
  renderMessage,
  renderSidePane,
  renderTextResponse,
  renderResponse,
  onMessagesChange,
  propAppSettings={},
}) {
  const [taskContext, updateContext] = React.useReducer(
    (oldContext, newContext) => {
      return { ...oldContext, ...newContext };
    },
    {currentAgentNames: []}
  );

  const [messages, addMessage] = React.useReducer(
    (previousMessages, newMessage) => {
      // we clear messages by sending false
      return newMessage === false ? [] : [...previousMessages, newMessage];
    },
    []
  );

  const initialAppSettings = {
    volume: 1, 
    isReview: false, 
    isCoverPage: false, 
    numMessages: 0,
    useTurns: true,
    ...propAppSettings
  };
  const [appSettings, setAppSettings] = React.useReducer(
    (prevSettings, newSettings) => Object.assign({}, prevSettings, newSettings),
    initialAppSettings
  );
  const [inputMode, setInputMode] = React.useState(INPUT_MODE.WAITING);


  React.useEffect(() => {
    if (onMessagesChange) {
      onMessagesChange(messages);
    }
    setAppSettings({ numMessages: messages.length });
  }, [messages]);

  function playNotifSound() {
    let audio = new Audio("./notif.mp3");
    audio.volume = appSettings.volume;
    audio.play();
  }

  function trackAgentName(agentName) {
    const previouslyTrackedNames = taskContext.currentAgentNames || {};
    const newAgentName = { [agentId]: agentName, [agentName]: agentName };
    const currentAgentNames = { ...previouslyTrackedNames, ...newAgentName };
    updateContext({ currentAgentNames: currentAgentNames });
  }

  function handleStateUpdate(state) {
    const {
      agent_display_name,
      live_update_requested,
      ...remainingState
    } = state;
    if (agent_display_name) {
      trackAgentName(agent_display_name);
    }
    if (remainingState.task_done) {
      setInputMode(INPUT_MODE.DONE);
    } else if (live_update_requested === true) {
      setInputMode(INPUT_MODE.READY_FOR_INPUT);
      if (appSettings.useTurns) {
        playNotifSound();
      }
    } else if (live_update_requested === false) {
      setInputMode(INPUT_MODE.WAITING);
    }
    if (Object.keys(remainingState).length > 0) {
      updateContext(remainingState);
    }
  }

  let mephistoProps = useMephistoLiveTask({
    onStatusUpdate: ({ status }) => {
      if (
        [
          AGENT_STATUS.DISCONNECT,
          AGENT_STATUS.RETURNED,
          AGENT_STATUS.EXPIRED,
          AGENT_STATUS.TIMEOUT,
          AGENT_STATUS.PARTNER_DISCONNECT,
          AGENT_STATUS.MEPHISTO_DISCONNECT,
        ].includes(status)
      ) {
        setInputMode(INPUT_MODE.INACTIVE);
        updateContext({
          doneText: STATUS_TO_TEXT_MAP[status],
          task_done: status == AGENT_STATUS.PARTNER_DISCONNECT,
        });
      }
    },
    onLiveUpdate: (message) => {
      if (message.task_data !== undefined) {
        handleStateUpdate(message.task_data);
      }
      if (message.text !== undefined) {
        addMessage(message);
      }

      // For handling reconnected packets and properly updating state
      // during turns.
      if (
        taskContext.currentAgentNames &&
        message.id in taskContext.currentAgentNames &&
        appSettings.useTurns
      ) {
        // This was our own message, so update to not requesting
        handleStateUpdate({ live_update_requested: false });
      }
    },
  });

  let {
    blockedReason,
    blockedExplanation,
    taskConfig,
    isPreview,
    previewHtml,
    isLoading,
    agentId,
    handleSubmit,
    connect,
    destroy,
    sendLiveUpdate,
    isOnboarding,
    agentStatus,
  } = mephistoProps;

  React.useEffect(() => {
    if (agentId) {
      console.log("connecting...");
      connect(agentId);
    }
  }, [agentId]);

  React.useEffect(() => {
    if (isOnboarding && agentStatus === AGENT_STATUS.WAITING) {
      handleSubmit();
    }
  }, [isOnboarding, agentStatus]);

  React.useEffect(() => {
    // clear messages when onboarding changes status
    addMessage(false);
  }, [isOnboarding])

  const handleMessageSend = React.useCallback(
    (message) => {
      message = {
        ...message,
        id: agentId,
        episode_done: taskContext?.task_done || false,
      };
      return sendLiveUpdate(message)
        .then(addMessage)
        .then(() => {
          if (appSettings.useTurns) {
            handleStateUpdate({ live_update_requested: false });
          }
        });
    },
    [agentId, taskContext?.task_done, addMessage, setInputMode]
  );

  if (blockedReason !== null) {
    return <h1>{blockedExplanation}</h1>;
  }
  if (isLoading) {
    return <div>Initializing...</div>;
  }
  if (isPreview) {
    if (!taskConfig.has_preview) {
      return <TaskPreviewView description={taskConfig.task_description} />;
    }
    if (previewHtml === null) {
      return <div>Loading...</div>;
    }
    return <div dangerouslySetInnerHTML={{ __html: previewHtml }} />;
  }
  if (isOnboarding) {
    return <OnboardingComponent 
      onboardingData={taskConfig.onboarding_data} 
      annotationBuckets={taskConfig.annotation_buckets} 
      annotationQuestion={taskConfig.annotation_question} 
      onSubmit={(dat) => {
        handleMessageSend({text: '', task_data: dat});
      }} 
    />;
  }

  return (
    <MephistoContext.Provider value={mephistoProps}>
      <AppContext.Provider
        value={{
          taskContext,
          appSettings,
          setAppSettings,
          onTaskComplete: () => {
            destroy();
            handleSubmit({});
          },
        }}
      >
        <div className="container-fluid" id="ui-container" style={{'display': "block"}}>
          <BaseFrontend
            inputMode={inputMode}
            messages={messages}
            onMessageSend={handleMessageSend}
            renderMessage={renderMessage}
            renderSidePane={renderSidePane}
            renderTextResponse={renderTextResponse}
            renderResponse={renderResponse}
          />
        </div>
      </AppContext.Provider>
    </MephistoContext.Provider>
  );
}

function TaskPreviewView({ description }) {
  return (
    <div className="preview-screen">
      <div
        dangerouslySetInnerHTML={{
          __html: description,
        }}
      />
    </div>
  );
}

export { CustomOnboardingChatApp, AppContext };