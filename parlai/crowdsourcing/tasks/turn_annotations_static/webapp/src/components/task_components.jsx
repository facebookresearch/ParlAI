/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { ErrorBoundary } from './error_boundary.jsx';
import { Checkboxes } from './checkboxes.jsx';

// HACK global variable (fix at some point)
// Array of arrays (by subtask and then by turn)
window.workerAnswers = [];

var showDisableCssNextButton = function () {
  // Have to disable the Next button in CSS b/c if React doesn't allow you to do
  // it in JS Note: person can actually click button even though it doesn't
  // look like they can! HACK: This is a hack; should tie it to state really
  document.getElementById('submit-button').style = 'cursor:not-allowed;background-color: #c8c8c8 !important;';
}

var showEnabledCssNextButton = function () {
  document.getElementById('submit-button').style = '';
}

var validateUserInput = function (subtaskData) {
  for (var i = 0; i < subtaskData.length; i++) {
    if (subtaskData[i].agent_idx == 1) {
      // only "bot" utterances (could be self-chat) have checkboxes
      var checkboxes = document.getElementsByName('checkbox_group_' + i);
      if (checkboxes.length > 0) {
        var atLeastOneAnswerChecked = false
        for (var j = 0; j < checkboxes.length; j++) {
          if (checkboxes[j].checked) {
            atLeastOneAnswerChecked = true;
            break;
          }
        }
        if (!atLeastOneAnswerChecked) {
          return false;
        }
      }
    }
  }
  return true;
}

var handleUserInputUpdate = function (subtaskData) {
  // subtaskData: task data with utterances (both human and bot) 
  // This checks that at least one checkbox is checked in every turn
  // before enabling the Next button
  // Needed b/c conversations scroll off the screen
  // HACK: need a more elegant way to check if all turns have an answer 
  if (validateUserInput(subtaskData)) {
    showEnabledCssNextButton();
  } else {
    showDisableCssNextButton();
  }
}

var handleSubtaskSubmit = function (subtaskIndex, setIndex, numSubtasks, initialTaskData, annotationBuckets, mephistoSubmit) {
  // initialTaskData is the initial task data for this index
  console.log('In handleSubtaskSubmit for subtask: ' + subtaskIndex);
  if (!validateUserInput(initialTaskData)) {
    alert('Data does not meet basic quality standards. Cannot be submitted.');
    return;
  }
  var answersForSubtaskIndex = {
    'subtask_index': subtaskIndex,
    data: []
  };
  for (var i = 0; i < initialTaskData.length; i++) {
    var buckets = Object.keys(annotationBuckets.config);
    var answersForTurn = {
      'turn_idx': i,
      'text': initialTaskData[i].text,
      'agent_idx': initialTaskData[i].agent_idx,
      'other_metadata': initialTaskData[i].other_metadata
    };
    for (var j = 0; j < buckets.length; j++) {
      answersForTurn[buckets[j]] = null;
      var checkbox = document.getElementById(buckets[j] + '_' + i);
      if (checkbox) {
        answersForTurn[buckets[j]] = false;
        if (checkbox.checked) {
          // Won't have checkboxes for agent_idx != 1
          answersForTurn[buckets[j]] = true;
          // uncheck any checked boxes
          checkbox.checked = false;
        }
      }
    }
    var input = document.getElementById('input_reason_' + i);
    if (input) {
      answersForTurn['input_reason'] = input.value;
    }
    answersForSubtaskIndex.data.push(answersForTurn);
    // Need to also manually clear the reason DIV below checkboxes
    if (answersForTurn.agent_idx == 1) {
      var checkbox_i = document.getElementById('checkbox_description_' + i);
      if (checkbox_i) {
        checkbox_i.innerHTML = '';
      }
      var input_i = document.getElementById('input_reason_' + i);
      if (input_i) {
        input_i.value = '';
      }
    }
  }
  window.workerAnswers.push(answersForSubtaskIndex);
  document.getElementById('right-pane').scrollTo(0, 0);

  if (subtaskIndex == (numSubtasks - 1)) {
    if (window.workerAnswers.length != numSubtasks) {
      alert('Unable to submit task due to malformed data. Please contact requester.');
      return;
    }
    mephistoSubmit(window.workerAnswers);
  }
  showDisableCssNextButton();
  setIndex(subtaskIndex + 1); 
}

function LeftPane({ stretch = false, children }) {
  let pane_size = stretch ? "col-xs-12" : "col-xs-4";
  return <div className={pane_size + " left-pane"} style={{ float: 'left', width: '30%', padding: '10px', boxSizing: 'border-box', backgroundColor: '#f2f2f2', height: '100%', overflow: 'scroll' }}>{children}</div>;
}

function RightPane({ children }) {
  return <div id="right-pane" className="right-pane" style={{ float: 'left', width: '70%', padding: '20px', boxSizing: 'border-box', height: '100%', overflow: 'scroll' }}>{children}</div>;
}

function SubtaskCounter({ subtaskIndex, numSubtasks }) {
  var taskword = numSubtasks === 1 ? 'conversation' : 'conversations';
  if (subtaskIndex >= numSubtasks) {
    return (
      <div>
      <b>Congratulations! All {numSubtasks} {taskword} have completed.</b> <br />
    </div>
    )
  } else {
  return (
      <div>
        <b>You are currently at conversation: {subtaskIndex + 1} / {numSubtasks} </b> <br />
    After completing each, click [NEXT] button, which will be enabled below.
      </div>)
  }
}

function SubtaskSubmitButton({ subtaskIndex, numSubtasks, onSubtaskSubmit }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <button id="submit-button"
        style={{ cursor: 'not-allowed', backgroundColor: '#c8c8c8' }}
        className="button is-link"
        onClick={onSubtaskSubmit}
      >
        {subtaskIndex === (numSubtasks - 1) ? "Submit" : "Next"}
      </button>
    </div>
  )
}

function ChatMessage({ text, agentIdx, annotationQuestion, annotationBuckets, turnIdx, doAnnotateMessage, askReason, onUserInputUpdate }) {
  var extraElements = '';
  if (doAnnotateMessage) {
    extraElements = '';
    extraElements = (<span key={'extra_' + turnIdx}><br /><br />
      <span style={{ fontStyle: 'italic' }} >
        <span dangerouslySetInnerHTML={{ __html: annotationQuestion }}></span>
        <br />
        <Checkboxes turnIdx={turnIdx} annotationBuckets={annotationBuckets} askReason={askReason} onUserInputUpdate={onUserInputUpdate} />
      </span>
    </span>)
  }
  return (
    <div className={`alert ${agentIdx == 0 ? "alert-info" : "alert-warning"}`} style={{ float: `${agentIdx == 0 ? "right" : "left"}`, display: 'table', minWidth: `${agentIdx == 0 ? "30%" : "80%"}`, marginTop: `${turnIdx == 1 ? "40px" : "auto"}` }}>
      <span><b>{turnIdx % 2 == 0 ? 'YOU' : 'THEM'}:</b> {text}
        <ErrorBoundary>
          {extraElements}
        </ErrorBoundary>
      </span>
    </div>
  )
}

function ContentPane({ subtaskData, taskConfig, subtaskIndex, numSubtasks }) {
  var annotationQuestion = taskConfig.annotation_question;
  var annotationBuckets = taskConfig.annotation_buckets;
  var annotateLastUtteranceOnly = taskConfig.annotate_last_utterance_only;
  var askReason = taskConfig.ask_reason;
  if (subtaskData == undefined && subtaskIndex >= numSubtasks) {
    // This happens when index gets set to num subtasks + 1 after submitting
    return (<div>
      Thank you for submitting this HIT!
    </div>);
  }

  return (
    <div>
      {subtaskData.map(
        (m, idx) =>
          <div key={idx}>
            <div>
              <ChatMessage
                text={m.text}
                agentIdx={m.agent_idx}
                turnIdx={idx}
                annotationQuestion={annotationQuestion}
                annotationBuckets={annotationBuckets}
                doAnnotateMessage={m.do_annotate}
                askReason={askReason}
                onUserInputUpdate={() => handleUserInputUpdate(subtaskData)}
              />
            </div>
          </div>
      )}
    </div>
  )
}

function MainTaskComponent({ taskData, taskTitle, taskDescription, taskConfig, onSubmit }) {
  if (taskData == undefined) {
    return <div><p> Loading chats...</p></div>;
  }
  const [index, setIndex] = React.useState(0);

  return (
    <div style={{ margin: 0, padding: 0, height: '100%' }}>
      <LeftPane>
        <h4><span dangerouslySetInnerHTML={{ __html: taskTitle || 'Task Title Loading' }}></span></h4>
        <SubtaskCounter subtaskIndex={index} numSubtasks={taskData.length}></SubtaskCounter>

        <br />
        <span dangerouslySetInnerHTML={{ __html: taskDescription || 'Task Description Loading' }}></span>
        <SubtaskSubmitButton subtaskIndex={index} numSubtasks={taskData.length} onSubtaskSubmit={() => { handleSubtaskSubmit(index, setIndex, taskData.length, taskData[index], taskConfig.annotation_buckets, onSubmit); }} initialTaskData={taskData[index]}></SubtaskSubmitButton>
      </LeftPane>
      <RightPane>
        <ContentPane subtaskData={taskData[index]} taskConfig={taskConfig} subtaskIndex={index} numSubtasks={taskData.length} ></ContentPane>
      </RightPane>
      <div style={{ clear: 'both' }}>
      </div>
    </div>
  );
}

export { MainTaskComponent };
