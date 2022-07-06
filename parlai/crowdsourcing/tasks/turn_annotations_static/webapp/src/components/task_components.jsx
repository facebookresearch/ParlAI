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
import { ErrorBoundary } from './error_boundary.jsx';
import { Checkboxes } from './checkboxes.jsx';
import { FormControl } from 'react-bootstrap';

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

var validateFreetextResponse = function (response, charMin, wordMin, vowelMin) {
  // Requires the response to contain at least charMin characters, wordMin words, and vowelMin vowels.
  var charCount = response.length;
  var wordCount = response.split(' ').length;
  var numVowels = response.match(/[aeiou]/gi);
  if (charCount >= charMin && wordCount >= wordMin && numVowels && numVowels.length >= vowelMin) {
    return true;
  } else {
    return false;
  }
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
  // check response fields
  var responses = document.getElementsByName('input_response');
  if (responses.length > 0) {
    for (var j = 0; j < responses.length; j++) {
      if (!validateFreetextResponse(responses[j].value, 10, 2, 2)) {
        return false;
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

var getAnnotationBuckets = function (taskData, annotationBuckets) {	
  // return list of all the bucket ids
  var buckets = []	
  if (annotationBuckets !== null && 'config' in annotationBuckets){	
    buckets = Object.keys(annotationBuckets.config);	
  }	
  taskData.forEach(elem => {	
    if ('annotation_buckets' in elem) {	
      (Object.keys(elem.annotation_buckets.config)).forEach(bucketKey => {	
        if (!buckets.includes(bucketKey)) {	
          buckets.push(bucketKey)	
        }	
      })	
    }	
  })	
  return buckets	
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
    var answersForTurn = {
      'turn_idx': i,
      'text': initialTaskData[i].text,
      'agent_idx': initialTaskData[i].agent_idx,
      'other_metadata': initialTaskData[i].other_metadata
    };
    var buckets = getAnnotationBuckets(initialTaskData, annotationBuckets)
    if (buckets !== null && buckets.length > 0) {
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
    }
    var input = document.getElementById('input_reason_' + i);
    if (input) {
      answersForTurn['input_reason'] = input.value;
    }
    var response = document.getElementById('input_response_' + i);
    if (response) {
      answersForTurn['input_response'] = response.value;
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
      var response_i = document.getElementById('input_response_' + i);
      if (response_i) {
        response_i.value = '';
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

function ChatMessage({ text, agentIdx, annotationQuestion, annotationBuckets, turnIdx, doAnnotateMessage, askReason, responseField, speakerLabel, onUserInputUpdate }) {
  var extraElements = '';
  var responseInputElement = '';
  if (speakerLabel == null) {	
    speakerLabel = turnIdx % 2 == 0 ? 'YOU' : 'THEM'
  }	
  var speakerElements = (	
    <div>	
      <b>{speakerLabel}:</b> {text}	
    </div>	
  )	

  if (doAnnotateMessage) {
    if (annotationBuckets !== null) { 
      extraElements = (<span key={'extra_' + turnIdx}><br /><br />
        <span style={{ fontStyle: 'italic' }} >
          <span dangerouslySetInnerHTML={{ __html: annotationQuestion }}></span>
          <br />
          <Checkboxes turnIdx={turnIdx} annotationBuckets={annotationBuckets} askReason={askReason} onUserInputUpdate={onUserInputUpdate} />
        </span>
      </span>)
    }
    if (responseField) {
      responseInputElement = (
        <FormControl
        type="text"
        name="input_response"
        id={"input_response_" + turnIdx}
        style={{
            fontSize: "14px",
            resize: "none",
            marginBottom: "40px"
        }}
        onChange={(e) => {onUserInputUpdate();}}
        placeholder={"Please enter your response here"}
        onPaste={(e) => {e.preventDefault(); alert("Please do not copy and paste. You must manually respond to each message.")}}
        autoComplete="off"
      />
      )
    }
  }
  return (
    <div>
      <div className={`alert ${agentIdx == 0 ? "alert-info" : "alert-warning"}`} style={{ float: `${agentIdx == 0 ? "right" : "left"}`, display: 'table', minWidth: `${agentIdx == 0 ? "30%" : "80%"}`, marginTop: "auto" }}>
        <span>
          {speakerElements}
          <ErrorBoundary>
            {extraElements}
          </ErrorBoundary>
        </span>
      </div>
      {responseInputElement}
    </div>
  )
}

function ContentPane({ subtaskData, taskConfig, subtaskIndex, numSubtasks }) {
  var annotationQuestion = taskConfig.annotation_question;
  var annotationBuckets = taskConfig.annotation_buckets;
  var askReason = taskConfig.ask_reason;
  var responseField = taskConfig.response_field;
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
                annotationQuestion={ ('annotation_question' in m) ? m.annotation_question : annotationQuestion}
                annotationBuckets={ ('annotation_buckets' in m) ? m.annotation_buckets : annotationBuckets}
                doAnnotateMessage={m.do_annotate}
                askReason={askReason}
                responseField={responseField}
                speakerLabel={('speaker_label' in m) ? m.speaker_label : null}
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
