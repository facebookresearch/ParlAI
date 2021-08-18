/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import InfoPanel from "./InfoPanel.jsx";
import SearchPanel from "./SearchPanel.jsx";
import OnboardingSidePane from "./OnBoardingSidePane.jsx";

export default function SidePane({
  mephistoContext,
  appContext,
  searchResults,
  selected,
  handleSelect,
  setSearchQuery,
  onBoardingStep,
  isWizard,
  apprenticePersona }) {

  const { currentAgentNames } = appContext.taskContext;
  if (!currentAgentNames) {  // The task is not started yet.
    return <WaitForStart />;
  }

  if (onBoardingStep > 0) {
    return <OnboardingSidePane
      onBoardingStep={onBoardingStep}
      mephistoContext={mephistoContext}
      appContext={appContext}
      searchResults={searchResults}
      selected={selected}
      handleSelect={handleSelect}
      setSearchQuery={setSearchQuery}
    />;
  }

  // Hidding the search bar while the agents are choosing persona
  const searchPanel = (!apprenticePersona || apprenticePersona === "") ? null :
    (<div id="search-area">
      <SearchPanel mephistoContext={mephistoContext}
        searchResults={searchResults}
        selected={selected}
        handleSelect={handleSelect}
        setSearchQuery={setSearchQuery}
        isWizard={isWizard} />
    </div>)

  return (
    <div className="side-pane" style={{ width: "100%" }}>
      <div>
        <InfoPanel isWizard={isWizard} personaDesc={apprenticePersona} />
      </div>
      {searchPanel}
    </div>
  )
}

function WaitForStart() {
  return <div>
    <strong>Please wait!</strong>
    <p>System is adding the partner and setting up the service.</p>
    <b>Matching may take up to 15 minutes (based on the number of online people). Please do not leave the chat.</b>
  </div>
}