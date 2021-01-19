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
import { FormControl } from 'react-bootstrap';

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

function SubtaskSubmitButton({
  subtaskIndex,
  numSubtasks,
  onSubtaskSubmit,
  disabled,
}) {
  return (
    <div style={{ textAlign: 'center' }}>
      <button id="submit-button"
        style={
          disabled ? {cursor: 'not-allowed', backgroundColor: '#c8c8c8'} : {}
        }
        className="button is-link"
        onClick={onSubtaskSubmit}
        disabled={disabled}
      >
        {subtaskIndex === (numSubtasks - 1) ? "Submit" : "Next"}
      </button>
    </div>
  );
}

class ChatMessage extends React.Component {
  // description below checkboxes is updated based on last selection
  state = {
    lastSelected: null,
    lastSubtaskIdx: 0,
  };
  constructor(props) {
    super(props);
  }

  static getDerivedStateFromProps(props, current_state) {
    // clear description when navigating to next task
    if (props.subtaskIdx != current_state.lastSubtaskIdx) {
      return {
        lastSelected: null,
        lastSubtaskIdx: props.subtaskIdx,
      }
    }
    return null
  }

  setLastSelected = (checkbox) => {
    this.setState({lastSelected: checkbox});
  }

  setLastSubtaskIdx = (subtaskIdx) => {
    this.setState({lastSubtaskIdx: subtaskIdx});
  }

  handleResponseFieldChange = evt => {
    this.props.setResponse(this.props.turnIdx, evt.target.value);
  };

  render() {
    const {
      annotationQuestion,
      annotationBuckets,
      turnIdx,
      turnAnswers,
      doAnnotateMessage,
      askReason,
      askResponse,
      setChecked,
      setReason,
    } = this.props;
    const {
      lastSelected,
    } = this.state;
    var extraElements = '';
    var responseInputElement = '';
    if (doAnnotateMessage) {
      if (annotationBuckets !== null) { 
        extraElements = (
          <span key={'extra_' + turnIdx}>
            <br />
            <br />
            <span style={{fontStyle: 'italic'}}>
              {annotationQuestion}
              <br />
              <Checkboxes
                turnIdx={turnIdx}
                turnAnswers={turnAnswers}
                annotationBuckets={annotationBuckets}
                askReason={askReason}
                lastSelected={lastSelected}
                setLastSelected={this.setLastSelected}
                setChecked={setChecked}
                setReason={setReason}
              />
            </span>
          </span>
        );
      }
      if (askResponse !== null) {
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
          onChange={this.handleResponseFieldChange}
          value={turnAnswers.response}
          placeholder={"Please enter your response here"}
          onPaste={(e) => {e.preventDefault(); alert("Please do not copy and paste. You must manually respond to each message.")}}
          autoComplete="off"
        />
        )
      }
    }
    const agentIdx = turnAnswers.agent_idx;
    return (
      <div
        className={`alert ${agentIdx == 0 ? 'alert-info' : 'alert-warning'}`}
        style={{
          float: `${agentIdx == 0 ? 'right' : 'left'}`,
          display: 'table',
          minWidth: `${agentIdx == 0 ? '30%' : '80%'}`,
          marginTop: `${turnIdx == 1 ? '40px' : 'auto'}`,
        }}>
        <span>
          <b>{turnIdx % 2 == 0 ? 'YOU' : 'THEM'}:</b> {turnAnswers.text}
          <ErrorBoundary>{extraElements}</ErrorBoundary>
        </span>
      </div>
    );
  }
}

function ContentPane({
  subtaskData,
  taskConfig,
  subtaskIndex,
  numSubtasks,
  setChecked,
  setReason,
  setResponse,
}) {
  var annotationQuestion = taskConfig.annotation_question;
  var annotationBuckets = taskConfig.annotation_buckets;
  var askReason = taskConfig.ask_reason;
  if (subtaskData == undefined && subtaskIndex >= numSubtasks) {
    // This happens when index gets set to num subtasks + 1 after submitting
    return <div>Thank you for submitting this HIT!</div>;
  }
  return (
    <div>
      {subtaskData.map((m, idx) => (
        <div key={idx}>
          <div>
            <ChatMessage
              subtaskIdx={subtaskIndex}
              turnIdx={idx}
              turnAnswers={m}
              annotationQuestion={annotationQuestion}
              annotationBuckets={annotationBuckets}
              doAnnotateMessage={m.do_annotate}
              askReason={askReason}
              askResponse={taskConfig.response_field}
              setChecked={setChecked}
              setReason={setReason}
              setResponse={setResponse}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// type Buckets = {
//   [bucket: string]: ?boolean,
// };

// type TurnAnswers = {
//   text: string,
//   agent_idx: number,
//   other_metadata: any,
//   buckets: Buckets,
//   do_annotate: boolean,
//   input_reason: ?string,
//   response: ?string,
// };

// type SubtaskAnswers = Array<TurnAnswers>;
// type WorkerAnswers = Array<SubtaskAnswers>;

// type MainTaskComponentProps = {
//   taskData: TaskData,
//   taskTitle: string,
//   taskDescription: string,
//   taskConfig: TaskConfig,
//   onSubmit: WorkerAnswers => void,
// };
// type MainTaskComponentState = {
//   index: number,
//   answers: WorkerAnswers,
// };

class MainTaskComponent extends React.Component {
  state = {
    index: 0,
    answers: this._initAnswersFromTaskData(),
  };
  constructor(props) {
    super(props);
  }

  _initAnswersFromTaskData() {
    var answers = [];
    for (var i = 0; i < this.props.taskData.length; i++) {
      var answersForSubtaskIndex = [];
      const subTaskData = this.props.taskData[i];
      for (var j = 0; j < subTaskData.length; j++) {
        var answersForTurn = {
          text: subTaskData[j].text,
          agent_idx: subTaskData[j].agent_idx,
          other_metadata: subTaskData[j].other_metadata,
          buckets: {},
          do_annotate: subTaskData[j].do_annotate,
          input_reason: null,
          response: null,
        };
        Object.keys(this.props.taskConfig.annotation_buckets.config).forEach(
          b => {
            answersForTurn.buckets[b] = null;
          },
        );
        answersForSubtaskIndex[j] = answersForTurn;
      }
      answers[i] = answersForSubtaskIndex;
    }
    return answers;
  }

  validateUserInput() {
    console.log('Validating user input');
    // if task is over, user should not be able to submit anything else
    if (this.state.index >= this.state.answers.length) {
      console.log('Task is over: ' + this.state.answers.length + ' vs ' + this.state.index);
      return false;
    }
    // check that all turns have answers
    const subtaskData = this.state.answers[this.state.index];
    for (var i = 0; i < subtaskData.length; i++) {
      const uttAnswer = subtaskData[i];
      if (uttAnswer.agent_idx == 1 && uttAnswer.do_annotate) {
        // only "bot" utterances (could be self-chat) have checkboxes
        var atLeastOneAnswerChecked = false;
        Object.keys(uttAnswer.buckets).forEach(b => {
          atLeastOneAnswerChecked =
            atLeastOneAnswerChecked || uttAnswer.buckets[b];
        });
        if (!atLeastOneAnswerChecked) {
          console.log('No answers checked');
          return false;
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
    console.log('Input valid!');
    return true;
  }

  handleSubtaskSubmit = () => {
    const subtaskIndex = this.state.index;
    console.log('In handleSubtaskSubmit for subtask: ' + subtaskIndex);
    if (!this.validateUserInput()) {
      alert('Data does not meet basic quality standards. Cannot be submitted.');
      return;
    }

    if (subtaskIndex == this.props.taskData.length - 1) {
      this.props.onSubmit(this.state.answers);
    }
    this.setState(prevState => ({index: prevState.index + 1}));
  };

  setChecked = (turnIdx, bucket, value) => {
    var answers = this.state.answers;
    answers[this.state.index][turnIdx].buckets[bucket] = value;
    this.setState({answers});
  };

  setReason = (turnIdx, reason) => {
    var answers = this.state.answers;
    answers[this.state.index][turnIdx].input_reason = reason;
    this.setState({answers});
  };

  setResponse = (turnIdx, response) => {
    var answers = this.state.answers;
    answers[this.state.index][turnIdx].response = response;
    this.setState({answers});
  }

  render() {
    const {taskData, taskTitle, taskDescription, taskConfig} = this.props;
    const {index, answers} = this.state;
    if (taskData == undefined) {
      return <div><p> Loading chats...</p></div>;
    }
    const task_title = taskTitle || 'Task Title Loading';
    const task_description = taskDescription || 'Task Description Loading';
    return (
      <div style={{ margin: 0, padding: 0, height: '100%' }}>
        <LeftPane>
          <h4>{task_title}</h4>
          <SubtaskCounter subtaskIndex={index} numSubtasks={taskData.length}></SubtaskCounter>

          <br />
          <span dangerouslySetInnerHTML={{ __html: task_description}}></span>
          <SubtaskSubmitButton 
            subtaskIndex={index} 
            numSubtasks={taskData.length} 
            onSubtaskSubmit={this.handleSubtaskSubmit}
            disabled={!this.validateUserInput()} />
        </LeftPane>
        <RightPane>
          <ContentPane 
            subtaskData={answers[index]}
            taskConfig={taskConfig}
            subtaskIndex={index}
            numSubtasks={taskData.length}
            setChecked={this.setChecked}
            setReason={this.setReason}
            setResponse={this.setResponse} />
        </RightPane>
        <div style={{ clear: 'both' }}>
        </div>
      </div>
    );
  }
}

export { MainTaskComponent };
