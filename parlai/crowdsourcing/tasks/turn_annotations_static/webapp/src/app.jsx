/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import { BaseFrontend, LoadingScreen } from "./components/core_components.jsx";
import { useMephistoTask } from "mephisto-task";

/* ================= Application Components ================= */

function MainApp() {
  const {
    blockedReason,
    blockedExplanation,
    isPreview,
    isLoading,
    initialTaskData,
    taskConfig,
    handleSubmit,
    isOnboarding,
  } = useMephistoTask();
  console.log('Entering MainApp');
  if (blockedReason !== null) {
    return (
      <section className="hero is-medium is-danger">
        <div class="hero-body">
          <h2 className="title is-3">{blockedExplanation}</h2>{" "}
        </div>
      </section>
    );
  }
  if (isLoading) {
    return <LoadingScreen />;
  }

  if (isPreview) {
    return (
      <section className="hero is-medium is-link">
        <div class="hero-body">
          <h3><span dangerouslySetInnerHTML={{ __html: taskConfig.task_title || 'Task Title Loading' }}></span></h3>
          <br />
          <span dangerouslySetInnerHTML={{ __html: taskConfig.task_description || 'Task Description Loading' }}></span>
        </div>
      </section>
    );
  }

  return (
    <div style={{ margin:0, padding:0, height:'100%' }}>
      <BaseFrontend
        taskData={initialTaskData}
        taskConfig={taskConfig}
        onSubmit={handleSubmit}
        isOnboarding={isOnboarding}
      />
    </div>
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
