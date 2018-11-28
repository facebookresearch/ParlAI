/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import ReactDOM from 'react-dom';
import {FormControl, Button} from 'react-bootstrap';

var component_list = null; // Will fill this in at the bottom
var CustomComponents = {};

function getCorrectComponent(component_name, agent_id) {
  if (CustomComponents[component_name] !== undefined) {
    if (CustomComponents[component_name][agent_id] !== undefined) {
      return CustomComponents[component_name][agent_id];
    } else if (CustomComponents[component_name]['default'] !== undefined) {
      return CustomComponents[component_name]['default'];
    }
  }
  return component_list[component_name][1];
}

class ChatMessage extends React.Component {
  render() {
    let float_loc = 'left';
    let alert_class = 'alert-warning';
    if (this.props.is_self) {
      float_loc = 'right';
      alert_class = 'alert-info';
    }
    return (
      <div className={"row"} style={{'marginLeft': '0', 'marginRight': '0'}}>
        <div
          className={"alert " + alert_class} role="alert"
          style={{'float': float_loc, 'display': 'table'}}>
          <span style={{'fontSize': '16px'}}>
            <b>{this.props.agent_id}</b>: {this.props.message}
          </span>
        </div>
      </div>
    );
  }
}

class MessageList extends React.Component {
  makeMessages() {
    let agent_id = this.props.agent_id;
    let messages = this.props.messages;
    // Handles rendering messages from both the user and anyone else
    // on the thread - agent_ids for the sender of a message exist in
    // the m.id field.
    let XChatMessage = getCorrectComponent('XChatMessage', this.props.v_id);
    return messages.map(
      m => <XChatMessage
        key={m.message_id}
        is_self={m.id == agent_id}
        agent_id={m.id}
        message={m.text}
        context={m.context}
        message_id={m.message_id}/>
    );
  }

  render () {
    return (
      <div id="message_thread" style={{'width': '100%'}}>
        {this.makeMessages()}
      </div>
    );
  }
}

class ConnectionIndicator extends React.Component {
  render () {
    let indicator_style = {
      'position': 'absolute', 'top': '5px', 'right': '10px',
      'opacity': '1', 'fontSize': '11px', 'color': 'white'
    };
    let text = '';
    switch (this.props.socket_status) {
      case 'connected':
        indicator_style['background'] = '#5cb85c';
        text = 'connected';
        break;
      case 'reconnecting_router':
        indicator_style['background'] = '#f0ad4e';
        text = 'reconnecting to router';
        break;
      case 'reconnecting_server':
        indicator_style['background'] = '#f0ad4e';
        text = 'reconnecting to server';
        break;
      case 'disconnected_server':
      case 'disconnected_router':
      default:
        indicator_style['background'] = '#d9534f';
        text = 'disconnected';
        break;
    }

    return (
      <button
        id="connected-button"
        className="btn btn-lg"
        style={indicator_style}
        disabled={true} >
          {text}
      </button>
    );
  }
}

class Hourglass extends React.Component {
  render () {
    // TODO move to CSS document
    let hourglass_style = {
      'marginTop': '-1px', 'marginRight': '5px',
      'display': 'inline', 'float': 'left'
    };

    // TODO animate?
    return (
      <div id="hourglass" style={hourglass_style}>
        <span
          className="glyphicon glyphicon-hourglass"
          aria-hidden="true" />
      </div>
    );
  }
}

class WaitingMessage extends React.Component {
  render () {
    let message_style = {
      float: 'left', display: 'table', 'backgroundColor': '#fff'
    };
    let text = 'Waiting for the next person to speak...'
    if (this.props.world_state == 'waiting') {
      text = 'Waiting to pair with a task...'
    }
    return (
      <div
        id="waiting-for-message"
        className="row"
        style={{'marginLeft': '0', 'marginRight': '0'}}>
          <div
            className="alert alert-warning"
            role="alert"
            style={message_style}>
            <Hourglass />
            <span style={{'fontSize': '16px'}}>
              {text}
            </span>
          </div>
      </div>
    );
  }
}

class ChatPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = {chat_height: this.getChatHeight()}
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.message_count != prevProps.message_count) {
      $('div#right-top-pane').animate({
        scrollTop: $('div#right-top-pane').get(0).scrollHeight
      }, 500);
    }
  }

  getChatHeight() {
    let entry_pane = $('div#right-bottom-pane').get(0);
    let bottom_height = 90;
    if (entry_pane !== undefined) {
      bottom_height = entry_pane.scrollHeight;
    }
    return this.props.frame_height - bottom_height;
  }

  render () {
    let v_id = this.props.v_id;
    let XMessageList = getCorrectComponent('XMessageList', v_id);
    let XWaitingMessage = getCorrectComponent('XWaitingMessage', v_id);

    // TODO move to CSS
    let chat_style = {
      'width': '100%', 'paddingTop': '60px',
      'paddingLeft': '20px', 'paddingRight': '20px',
      'paddingBottom': '20px', 'overflowY': 'scroll'
    };

    window.setTimeout(() => {
      if (this.getChatHeight() != this.state.chat_height) {
        this.setState({chat_height: this.getChatHeight()});
      }
    }, 10);

    chat_style['height'] = (this.state.chat_height) + 'px'

    let wait_message = null;
    if (this.props.chat_state == 'waiting') {
      wait_message = <XWaitingMessage {...this.props}/>;
    }

    return (
      <div id="right-top-pane" style={chat_style}>
        <XMessageList {...this.props} />
        <ConnectionIndicator {...this.props} />
        {wait_message}
      </div>
    );
  }
}

class IdleResponse extends React.Component {
  render() {
    return (
      <div
        id="response-type-idle"
        className="response-type-module" />
    );
  }
}

class DoneButton extends React.Component {
  // This component is responsible for initiating the click
  // on the mturk form's submit button.
  render() {
    return (
      <button
        id="done-button" type="button"
        className="btn btn-primary btn-lg"
        onClick={() => this.props.allDoneCallback()}>
          <span
            className="glyphicon glyphicon-ok-circle"
            aria-hidden="true" /> Done with this HIT
      </button>
    );
  }
}

class DoneResponse extends React.Component {
  render () {
    let v_id = this.props.v_id;
    let XDoneButton = getCorrectComponent('XDoneButton', v_id);

    let inactive_pane = null;
    if (this.props.done_text) {
      inactive_pane = (
        <span
          id="inactive"
          style={{'fontSize': '14pt', 'marginRight': '15px'}}>
          {this.props.done_text}
        </span>
      );
    }
    // TODO maybe move to CSS?
    let pane_style = {
      'paddingLeft': '25px',
      'paddingTop': '20px',
      'paddingBottom': '20px',
      'paddingRight': '25px',
      'float': 'left'
    };
    let done_button = <XDoneButton {...this.props} />;
    if (!this.props.task_done) {
      done_button = null;
    }
    return (
      <div id="response-type-done" className="response-type-module"
        style={pane_style}>
          {inactive_pane}
          {done_button}
      </div>
    );
  }
}

class TextResponse extends React.Component {
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
        onChange={(e) => this.setState({textval: e.target.value})}
        disabled={!this.props.active}/>
    );

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

class ResponsePane extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XDoneResponse = getCorrectComponent('XDoneResponse', v_id);
    let XTextResponse = getCorrectComponent('XTextResponse', v_id);
    let XIdleResponse = getCorrectComponent('XIdleResponse', v_id);

    let response_pane = null;
    switch (this.props.chat_state) {
      case 'done':
      case 'inactive':
        response_pane = <XDoneResponse
          {...this.props}
        />;
        break;
      case 'text_input':
      case 'waiting':
        response_pane = <XTextResponse
          {...this.props}
          active={this.props.chat_state == 'text_input'}
        />;
        break;
      case 'idle':
      default:
        response_pane = <XIdleResponse {...this.props} />;
        break;
    }

    return (
      <div
        id="right-bottom-pane"
        style={{width: '100%', 'backgroundColor': '#eee'}}>
        {response_pane}
      </div>
    );
  }
}

class RightPane extends React.Component {
  render () {
    let v_id = this.props.v_id;
    let XChatPane = getCorrectComponent('XChatPane', v_id);
    let XResponsePane = getCorrectComponent('XResponsePane', v_id);

    // TODO move to CSS
    let right_pane = {
      'minHeight': '100%', 'display': 'flex', 'flexDirection': 'column',
      'justifyContent': 'spaceBetween'
    };

    return (
      <div id="right-pane" style={right_pane}>
        <XChatPane message_count={this.props.messages.length} {...this.props} />
        <XResponsePane {...this.props} />
      </div>
    );
  }
}

class LeftPane extends React.Component {
  render () {
    let frame_height = this.props.frame_height;
    let frame_style = {
      height: frame_height + 'px',
      'backgroundColor': '#dff0d8',
      padding: '30px',
      overflow: 'auto'
    };
    let pane_size = this.props.full ? 'col-xs-12' : 'col-xs-4';
    // TODO pull from templating variable
    let header_text = "Live Chat";
    let task_desc = this.props.task_description || 'Task Description Loading';
    return (
      <div id="left-pane" className={pane_size} style={frame_style}>
          <h1>{header_text}</h1>
          <hr style={{'borderTop': '1px solid #555'}} />
          <span
            id="task-description" style={{'fontSize': '16px'}}
            dangerouslySetInnerHTML={{__html: task_desc}}
          />
      </div>
    );
  }
}

class ContentLayout extends React.Component {
  render () {
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XRightPane = getCorrectComponent('XRightPane', v_id);
    return (
      <div className="row" id="ui-content">
        <XLeftPane
          full={false}
          {...this.props}
        />
        <XRightPane {...this.props} />
      </div>
    );
  }
}

class BaseFrontend extends React.Component {
  render () {
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XContentLayout = getCorrectComponent('XContentLayout', v_id);

    let content = null;
    if (this.props.is_cover_page) {
      content = (
        <div className="row" id="ui-content">
          <XLeftPane
            full={true}
            {...this.props}
          />
        </div>
      );
    } else if (this.props.initialization_status == 'initializing') {
      content = <div id="ui-placeholder">Initializing...</div>;
    } else if (this.props.initialization_status == 'websockets_failure') {
      content = <div id="ui-placeholder">
        Sorry, but we found that your browser does not support WebSockets.
        Please consider updating your browser to a newer version and check
        this HIT again.
      </div>;
    } else if (this.props.initialization_status == 'failed') {
      content = <div id="ui-placeholder">
        Unable to initialize. We may be having issues with our chat servers.
        Please refresh the page, or if that isn't working return the HIT and
        try again later if you would like to work on this task.
      </div>;
    } else {
      content = <XContentLayout {...this.props} />;
    }
    return (
      <div className="container-fluid" id="ui-container">
        {content}
      </div>
    );
  }
}

function setCustomComponents(new_components) {
  CustomComponents = new_components;
}

component_list = {
  'XContentLayout': ['ContentLayout', ContentLayout],
  'XLeftPane': ['LeftPane', LeftPane],
  'XRightPane': ['RightPane', RightPane],
  'XResponsePane': ['ResponsePane', ResponsePane],
  'XTextResponse': ['TextResponse', TextResponse],
  'XDoneResponse': ['DoneResponse', DoneResponse],
  'XIdleResponse': ['IdleResponse', IdleResponse],
  'XDoneButton': ['DoneButton', DoneButton],
  'XChatPane': ['ChatPane', ChatPane],
  'XWaitingMessage': ['WaitingMessage', WaitingMessage],
  'XMessageList': ['MessageList', MessageList],
  'XChatMessage': ['ChatMessage', ChatMessage]
};

export {
  // Original Components
  ChatMessage, MessageList, ConnectionIndicator, Hourglass, WaitingMessage,
  ChatPane, IdleResponse, DoneButton, DoneResponse, TextResponse, ResponsePane,
  RightPane, LeftPane, ContentLayout, BaseFrontend,
  // Functions to update current components
  setCustomComponents
};
