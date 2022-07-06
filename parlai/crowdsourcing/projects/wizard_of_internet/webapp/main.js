/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import ReactDOM from "react-dom";
import "bootstrap-chat/styles.css";

import { FormControl, Button } from "react-bootstrap";
import { ChatApp, ChatMessage, INPUT_MODE } from "bootstrap-chat";

import SidePane from "./components/SidePane.jsx";
import { OnboardingSteps } from "./components/OnBoardingSidePane.jsx";
import valid_utterance from "./components/Moderator.js";

import "./components/styles.css";

function isOnboardingMessage(message) {
  const sender_id = message["id"].toLocaleLowerCase();
  return sender_id.startsWith("onboarding");
}

function newMessageHandler(
  messages,
  setApprenticePersona,
  setOnBoardingStep,
  updateSearchResults,
  updateSelectedSearchResults) {

  if ((!messages) || (messages.length < 1)) {
    return;
  }

  function resetSelected(search_results) {
    var selected = [[false]];
    for (let doc_id = 0; doc_id < search_results.length; doc_id++) {
      const doc = search_results[doc_id];
      var selected_sentences = [];
      for (let sentence_id = 0; sentence_id < doc.content.length; sentence_id++) {
        selected_sentences.push(false);
      }
      selected.push(selected_sentences);
    }
    updateSelectedSearchResults(selected);
  }

  const msg = messages[messages.length - 1]
  if (msg.id === "SearchAgent") {
    const searh_res = msg.task_data["search_results"];
    resetSelected(searh_res);
    updateSearchResults(searh_res);
  } else if (msg.id === "PersonaAgent") {
    const persona = msg['task_data']['apprentice_persona'];
    setApprenticePersona(persona);
  } else if (msg.id === "OnboardingBot") {
    const taskDataKey = 'task_data';
    const onboardingStepKey = 'on_boarding_step';
    if ((taskDataKey in msg) && (onboardingStepKey in msg[taskDataKey])) {
      const stepDuringOnboarding = msg['task_data'][onboardingStepKey];
      console.log("Setting onboarding step to ", stepDuringOnboarding);
      setOnBoardingStep(stepDuringOnboarding);
    }
  }
}

function isWizard(mephistoContext, appContext, onBoardingStep) {
  if ((onBoardingStep === OnboardingSteps.TRY_SEARCH) ||
    (onBoardingStep === OnboardingSteps.PERSONA_WIZARD)) {
    return true;
  }

  const { agentId } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;
  if (!currentAgentNames) {
    return false;
  }
  const agentName = currentAgentNames[agentId];
  const answer = (agentName === "Wizard") ? true : false;
  return answer;
}

function RenderChatMessage({ message, mephistoContext, appContext, setIsMinTurnsMet, isInMainTask }) {
  if (message.text === "") {
    return null;
  }

  // TODO: replace this hacky solution for removing remaining message from onboarding
  // with a better solution that purges them from the messages list.
  if (isInMainTask && isOnboardingMessage(message)) {
    return null;
  }

  const { agentId } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;
  const taskDataKey = 'task_data';
  const searchResultsKey = 'search_results';
  if ((taskDataKey in message) && (searchResultsKey in message[taskDataKey])) {
    // The received message comes from Search query: DO NOT ADD TO CHAT.
    return null;
  }

  const nTurnsKeys = "utterance_count";  // Because this is observed and comes from seen messages
  if (("task_data" in message) && (nTurnsKeys in message.task_data)) {
    const numTurns = message.task_data[nTurnsKeys];
    const minNumTurns = mephistoContext.taskConfig["minTurns"];
    if (numTurns > minNumTurns) {
      setIsMinTurnsMet(true);
    }
  }

  const isSelf = ((message.id === agentId) || (message.id in currentAgentNames));
  let shownName;
  if (isSelf) {
    shownName = "You";
  } else {
    if (["Wizard", "Apprentice"].includes(message.id)) {
      shownName = "Your Partner";
    } else {
      shownName = message.id;
    }
  }

  return (
    <div>
      <ChatMessage
        isSelf={isSelf}
        agentName={shownName}
        message={message.text}
        taskData={message.task_data}
        messageId={message.message_id}
      />
    </div>
  );
}

function CustomTextResponse({
  taskConfig,
  onMessageSend,
  active,
  searchQuery,
  setSearchQuery,
  searchResults,
  setSearchResults,
  selectedSearchResults,
  setSelectedSearchResults,
  isMinTurnMet,
  isWizard,
  isOnboarding,
}) {
  const [textValue, setTextValue] = React.useState("");
  const [sending, setSending] = React.useState(false);

  const inputRef = React.useRef();

  React.useEffect(() => {
    if (active && inputRef.current && inputRef.current.focus) {
      inputRef.current.focus();
    }
  }, [active]);

  const trySignalFinishChat = () => {
    if (active && !sending) {
      setSending(true);
      onMessageSend({ text: "", requested_finish: true }).then(
        () => { setSending(false); })
    }
  };

  function needSelection(selMatrix) {
    if (!isWizard) {
      return false;
    }
    for (var i = 0; i < selMatrix.length; i++) {
      for (var j = 0; j < selMatrix[i].length; j++) {
        if (selMatrix[i][j]) {
          return false
        }
      }
    }
    return true;
  }

  const tryMessageSend = React.useCallback(() => {
    if (needSelection(selectedSearchResults)) {
      alert("Please select an option from the left panel.")
      return;
    }
    if (textValue !== "" &&
      active &&
      !sending &&
      valid_utterance(textValue, searchResults, selectedSearchResults, isOnboarding, taskConfig)) {
      setSending(true);
      onMessageSend({
        timestamp: Date.now(),
        text: textValue,
        task_data: {
          search_query: searchQuery,
          text_candidates: searchResults,
          selected_text_candidates: selectedSearchResults,
        }
      }).then(() => {
        setTextValue("");
        setSearchQuery("");
        setSearchResults([]);
        setSelectedSearchResults([[false]]);
        setSending(false);
      });
    }
  }, [textValue, active, sending, onMessageSend, selectedSearchResults]);

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

  const finishButton = (isMinTurnMet) ?
    <Button
      className="btn btn-success submit-response"
      id="id_finish_button"
      disabled={!active || sending}
      onClick={() => trySignalFinishChat()}
    >
      Finish
    </Button> : null;

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
          placeholder="Enter your message here..."
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
        {finishButton}
      </div>
    </div>
  );
}

function MainApp() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedSearchResults, setSelectedSearchResults] = useState([[false]]);
  const [isMinTurnMet, setIsMinTurnsMet] = useState(false);
  const [apprenticePersona, setApprenticePersona] = useState("");
  const [onBoardingStep, setOnBoardingStep] = useState(OnboardingSteps.NOT_ONBOARDING);

  function handleSelect(loc) {
    const [doc_id, sentence_id] = loc;
    const new_selected = selectedSearchResults.slice();
    if ((doc_id === 0) && (sentence_id === 0)) {  // No sentce selected
      new_selected[0][0] = !new_selected[0][0];
      for (var i = 1; i < new_selected.length; i++) {
        for (var j = 0; j < new_selected[i].length; j++) {
          new_selected[i][j] = false;
        }
      }
    } else {  // Any other selected
      new_selected[0][0] = false;
      const prev_val = selectedSearchResults[doc_id][sentence_id];
      new_selected[doc_id][sentence_id] = !prev_val;
    }
    setSelectedSearchResults(new_selected);
  }

  return (
    <div>
      <ChatApp
        renderMessage={({ message, idx, mephistoContext, appContext }) => (
          <RenderChatMessage
            message={message}
            mephistoContext={mephistoContext}
            appContext={appContext}
            setIsMinTurnsMet={setIsMinTurnsMet}
            key={message.message_id + "-" + idx}
            isInMainTask={onBoardingStep === OnboardingSteps.NOT_ONBOARDING}
          />
        )}
        renderTextResponse={({ onMessageSend, inputMode, mephistoContext, appContext }) =>
        (<CustomTextResponse
          taskConfig={mephistoContext.taskConfig}
          onMessageSend={onMessageSend}
          active={inputMode === INPUT_MODE.READY_FOR_INPUT}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          searchResults={searchResults}
          setSearchResults={setSearchResults}
          selectedSearchResults={selectedSearchResults}
          setSelectedSearchResults={setSelectedSearchResults}
          isMinTurnMet={isMinTurnMet}
          isWizard={isWizard(mephistoContext, appContext, onBoardingStep)}
          isOnboarding={onBoardingStep !== OnboardingSteps.NOT_ONBOARDING}
        />)
        }
        onMessagesChange={(messages) => (
          newMessageHandler(messages,
            setApprenticePersona,
            setOnBoardingStep,
            setSearchResults,
            setSelectedSearchResults))}

        renderSidePane={({ mephistoContext, appContext }) => (
          <SidePane mephistoContext={mephistoContext}
            appContext={appContext}
            searchResults={searchResults}
            selected={selectedSearchResults}
            handleSelect={handleSelect}
            setSearchQuery={setSearchQuery}
            onBoardingStep={onBoardingStep}
            isWizard={isWizard(mephistoContext, appContext)}
            apprenticePersona={apprenticePersona} />
        )}
      />
    </div>
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
