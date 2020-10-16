/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import {
  TaskDescription,
  BaseFrontend,
} from "./components/core_components.jsx";
import { useMephistoTask, getBlockedExplanation } from "mephisto-task";

/* ================= Application Components ================= */

function MainApp() {
  const {
    blockedReason,
    taskConfig,
    isPreview,
    isLoading,
    initialTaskData,
    handleSubmit,
  } = useMephistoTask();

  if (blockedReason !== null) {
    return <h1>{getBlockedExplanation(blockedReason)}</h1>;
  }
  if (isPreview) {
    return <TaskDescription task_config={taskConfig} is_cover_page={true} />;
  }
  if (isLoading) {
    return <div>Initializing...</div>;
  }
  if (initialTaskData === null) {
    return <h1>Gathering data...</h1>;
  }

  return (
    <div>
      <BaseFrontend
        task_data={initialTaskData}
        task_config={taskConfig}
        onSubmit={handleSubmit}
      />
    </div>
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
