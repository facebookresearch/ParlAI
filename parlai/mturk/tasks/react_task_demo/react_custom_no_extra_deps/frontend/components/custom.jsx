/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';

// Create custom components
class EvaluatorIdleResponse extends React.Component {
  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      'paddingLeft': '25px',
      'paddingTop': '20px',
      'paddingBottom': '20px',
      'paddingRight': '25px',
      'float': 'left'
    };

    return (
      <div
        id="response-type-idle"
        className="response-type-module"
        style={pane_style}>
        <span>
          Pay attention to the conversation above, as you'll need to evaluate.
        </span>
      </div>
    );
  }
}

class NumericResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {'textval': ''};
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.active) {
      $("input#id_text_input").focus();
    }
  }

  tryMessageSend() {
    if (this.state.textval != '' && this.props.active) {
      this.props.onMessageSend(
        this.state.textval, {}, () => this.setState({textval: ''}));
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  updateValue(amount) {
    if ((amount != '' && isNaN(amount)) || amount < 0) {
      return;
    }
    amount = amount == '' ? 0 : amount;
    this.setState({textval: '' + amount})
  }

  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      'paddingLeft': '25px',
      'paddingTop': '20px',
      'paddingBottom': '20px',
      'paddingRight': '25px',
      'float': 'left',
      'width': '100%'
    };
    let input_style = {
      height: "50px", width: "100%", display: "block", float: 'left'
    };
    let submit_style = {
      'width': '100px', 'height': '100%', 'fontSize': '16px',
      'float': 'left', 'marginLeft': '10px', 'padding': '0px'
    };

    let text_input = (
      <FormControl
        type="text"
        id="id_text_input"
        style={{width: '80%', height: '100%', float: 'left', 'fontSize': '16px'}}
        value={this.state.textval}
        placeholder="Please enter here..."
        onKeyPress={(e) => this.handleKeyPress(e)}
        onChange={(e) => this.updateValue(e.target.value)}
        disabled={!this.props.active}/>
    );

    // TODO attach send message callback
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={this.state.textval == '' || !this.props.active}
        onClick={() => this.tryMessageSend()}>
          Send
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}>
          <div style={input_style}>
            {text_input}
            {submit_button}
          </div>
      </div>
    );
  }
}

class EvaluationResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {'textval': ''};
  }

  tryMessageSend(value) {
    if (this.props.active) {
      this.props.onMessageSend(
        value, {}, () => this.setState({textval: ''}));
    }
  }

  render() {
    // TODO maybe move to CSS?
    let pane_style = {
      'paddingLeft': '25px',
      'paddingTop': '20px',
      'paddingBottom': '20px',
      'paddingRight': '25px',
      'float': 'left',
      'width': '100%'
    };
    let input_style = {
      height: "50px", width: "100%", display: "block", float: 'left'
    };
    let submit_style = {
      'width': '100px', 'height': '100%', 'fontSize': '16px',
      'float': 'left', 'marginLeft': '10px', 'padding': '0px'
    };

    let reject_button = (
      <Button
        className="btn btn-danger"
        style={submit_style}
        id="id_reject_chat_button"
        disabled={!this.props.active}
        onClick={() => this.tryMessageSend('reject')}>
          Reject
      </Button>
    );

    let approve_button = (
      <Button
        className="btn btn-success"
        style={submit_style}
        id="id_approve_chat_button"
        disabled={!this.props.active}
        onClick={() => this.tryMessageSend('approve')}>
          Approve
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}>
          <div style={input_style}>
            {reject_button}
            {approve_button}
          </div>
      </div>
    );
  }
}

// Package components
var IdleResponseHolder = {
  // default: leave blank to use original default when no ids match
  'Evaluator': EvaluatorIdleResponse
}

var TextResponseHolder = {
  // default: leave blank to use original default when no ids match
  'Evaluator': EvaluationResponse,
  'Onboarding Evaluator': EvaluationResponse,
  'Answerer': NumericResponse,
  'Onboarding Answerer': NumericResponse
}

export default {
  // ComponentName: CustomReplacementComponentMap
  XTextResponse: TextResponseHolder,
  XIdleResponse: IdleResponseHolder
};
