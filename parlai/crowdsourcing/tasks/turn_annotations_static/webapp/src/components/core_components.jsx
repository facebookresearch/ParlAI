/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { OnboardingComponent } from './onboarding_components.jsx'; 
import { MainTaskComponent } from './task_components.jsx'; 

function LoadingScreen() {
  return <div>Loading...</div>;
}

function TaskFrontend({ taskData, taskConfig, isOnboarding, onSubmit }) {
  if (!taskData) {
    return <LoadingScreen />;
  }
  if (isOnboarding) {
    return <OnboardingComponent onboardingData={taskConfig.onboarding_data} annotationBuckets={taskConfig.annotation_buckets} annotationQuestion={taskConfig.annotation_question} onSubmit={onSubmit} />;
  }
  return (
    <MainTaskComponent taskData={taskData} taskTitle={taskConfig.task_title} taskDescription={taskConfig.task_description} taskConfig={taskConfig} onSubmit={onSubmit}></MainTaskComponent>
  );
}

export { LoadingScreen, TaskFrontend as BaseFrontend };
