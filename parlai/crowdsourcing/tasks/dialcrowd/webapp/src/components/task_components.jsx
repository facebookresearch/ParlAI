/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

 // NOTE: this frontend uses document accessors rather than React to control state,
 // and may not be compatible with some future Mephisto features

import React from "react";
import WorkerCategory from './dialcrowd/worker_category.js'

function MainTaskComponent({ taskData, taskConfig, onSubmit }) {
  if (taskData == undefined) {
    return <div><p> Loading chats...</p></div>;
  }

  var data_amount = taskConfig.category_data.length

  // inject duplicated tasks and golden task units{
  for (let i = 0; i < taskConfig.nUnitDuplicated; i++){
    var duplicate_data = taskData[Math.floor(Math.random()*taskData.length)];
    taskData.push({'id': data_amount + duplicate_data['id'], 'sentences': duplicate_data['sentences'], 'category': []})
  }

  data_amount = data_amount*2 + 1

  if (taskConfig.nUnitGolden > 0){
    for (let i = 0; i < taskConfig.nUnitGolden; i++){
      taskData.push({'id': data_amount, 'sentences': [taskConfig.dataGolden[i]['sentence']], 'category': []})
      data_amount += 1
    }
  }

  console.log(taskData)

  const [index, setIndex] = React.useState(0);
  return <WorkerCategory taskData={taskData} taskConfig={taskConfig} onSubmit={onSubmit}/>;
}

export { MainTaskComponent };
