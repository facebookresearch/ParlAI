/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import SearchPanel from "./SearchPanel.jsx";

// NOTE: these need to match ONBOARDING_STEPS dict in constants.py
export const OnboardingSteps = {
  NOT_ONBOARDING: 0,
  CHAT_INTERFACE: 1,
  TRY_SEARCH: 2,
  PERSONA_WIZARD: 3,
  PERSONA_APPRENTICE: 4,
  WAITING: 10
};

export default function OnboardingSidePane({ onBoardingStep, mephistoContext,
  searchResults, selected, handleSelect, setSearchQuery }) {
  let tutorialComponent = <div>Error: Unknown OnBoarding Step.</div>;
  switch (onBoardingStep) {
    case OnboardingSteps.CHAT_INTERFACE:
      tutorialComponent = <OnboardingSidePanel
        mephistoContext={mephistoContext}
        hideSearchBar={true}
        hidePersona={true}
      />;
      break;

    case OnboardingSteps.TRY_SEARCH:
      tutorialComponent = <OnboardingSidePanel
        mephistoContext={mephistoContext}
        searchResults={searchResults}
        selected={selected}
        handleSelect={handleSelect}
        setSearchQuery={setSearchQuery}
        hideSearchBar={false}
        blinkSearchBar={true}
        hidePersona={true}
        blinkPersona={false}
      />;
      break;

    case OnboardingSteps.PERSONA_WIZARD:
      tutorialComponent = <OnboardingSidePanel
        mephistoContext={mephistoContext}
        searchResults={searchResults}
        selected={selected}
        handleSelect={handleSelect}
        setSearchQuery={setSearchQuery}
        hideSearchBar={false}
        blinkSearchBar={false}
        hidePersona={false}
        blinkPersona={true}
      />;
      break;

    case OnboardingSteps.PERSONA_APPRENTICE:
      tutorialComponent = <OnboardingSidePanel
        mephistoContext={mephistoContext}
        hideSearchBar={true}
        hidePersona={false}
        blinkPersona={true}
      />;
      break;

    case OnboardingSteps.WAITING:
      tutorialComponent = <Waiting />;
      break;

    default:
      console.error("Unrecognized onboarding step " + onBoardingStep);
  }

  return (
    <div>
      {tutorialComponent}
    </div>

  )
}

function OnboardingSidePanel({
  mephistoContext,
  searchResults,
  selected,
  handleSelect,
  setSearchQuery,
  hideSearchBar,
  blinkSearchBar,
  hidePersona,
  blinkPersona }) {
  const peronaDescription = mephistoContext.taskConfig["onboardingPersona"];
  const SearchBar = hideSearchBar ? null
    : (<div id="search-area">
      <div className={blinkSearchBar ? "blinking" : ""}>
        <SearchPanel mephistoContext={mephistoContext}
          searchResults={searchResults}
          selected={selected}
          handleSelect={handleSelect}
          setSearchQuery={setSearchQuery}
          isWizard={true}
        />
      </div>
    </div>);
  return (
    <div className="side-pane" style={{ width: "100%" }}>
      <div id="info-bar">
        <TaskDescription />
        <Persona
          isBlinking={blinkPersona}
          personaDesc={(hidePersona) ? "" : peronaDescription}
        />
      </div>
      {SearchBar}
    </div>

  );
}

function Waiting() {
  return (
    <div id="info-bar">
      <div className={"info-pane " + "instruction-pane"}>
        <h2>Instruction.</h2>
        <p>
          Please wait while we pair you with other participants.
      </p>
      </div>
      <Persona
        isBlinking={false}
        personaDesc=""
      />
    </div>
  );
}

function Persona({ isBlinking, personaDesc }) {
  const blinkingStyle = (isBlinking === true) ? "blinking" : "non-blinking";
  const persona = (personaDesc === "") ? "" :
    (<div className={blinkingStyle}>
      <h3>Character description</h3>
      <h1><i>
        {personaDesc}
      </i></h1>
    </div>);
  return (
    <div className={"info-pane " + "persona-pane "}>
      {persona}
    </div>);
}

function TaskDescription() {
  return (
    <div className={"info-pane " + "instruction-pane"}>
      <h2>Instruction.</h2>
      <p>
        Our OnboardingBot is sending you instructions (see the right pane).
        Follow them to get familiar with this task and its environment,
        before we start the main task.
        </p>
    </div>);
}