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
import CustomComponents from './components/custom.jsx';
import SocketHandler from './components/socket_handler.jsx';
import {MTurkSubmitForm, allDoneCallback} from './components/mturk_submit_form.jsx';
import 'fetch';

/* ================= Utility functions ================= */

// Determine if the browser is a mobile phone
function isMobile() {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent);
}

// Sends a request to get the hit_config
function getHitConfig(callback_function) {
  $.ajax({
    url: '/get_hit_config',
    timeout: 3000 // in milliseconds
  }).retry({times: 10, timeout: 3000}).then(
    function(data) {
      if (callback_function) {
        callback_function(data);
      }
    }
  );
}

// Sees if the current browser supports WebSockets
function doesSupportWebsockets() {
  return !((bowser.msie && bowser.version < 10) ||
           (bowser.firefox && bowser.version < 11) ||
           (bowser.chrome && bowser.version < 16) ||
           (bowser.safari && bowser.version < 7) ||
           (bowser.opera && bowser.version < 12.1));
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
    return messages.map(
      m => <XChatMessage
        key={m.message_id}
        is_self={m.id == agent_id}
        agent_id={m.id}
        message={m.text}
        message_id={m.id}/>
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
              Waiting for the next person to speak...
            </span>
          </div>
      </div>
    );
  }
}

class ChatPane extends React.Component {
  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.message_count != prevProps.message_count) {
      $('div#right-top-pane').animate({
        scrollTop: $('div#right-top-pane').get(0).scrollHeight
      }, 500);
    }
  }

  render () {
    // TODO move to CSS
    let chat_style = {
      'width': '100%', 'paddingTop': '60px',
      'paddingLeft': '20px', 'paddingRight': '20px',
      'paddingBottom': '20px', 'overflowY': 'scroll'
    };

    chat_style['height'] = (this.props.frame_height - 90) + 'px'

    let wait_message = null;
    if (this.props.chat_state == 'waiting') {
      wait_message = <XWaitingMessage />;
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
        onClick={() => allDoneCallback()}>
          <span
            className="glyphicon glyphicon-ok-circle"
            aria-hidden="true" /> Done with this HIT
      </button>
    );
  }
}

class DoneResponse extends React.Component {
  render () {
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
    let done_button = <XDoneButton />;
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
        response_pane = <XIdleResponse />;
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
    return (
      <div className="row" id="ui-content">
        <XLeftPane
          task_description={this.props.task_description}
          full={false}
          frame_height={this.props.frame_height}
        />
        <XRightPane {...this.props} />
      </div>
    );
  }
}

class BaseFrontend extends React.Component {
  render () {
    let content = null;
    if (this.props.is_cover_page) {
      content = (
        <div className="row" id="ui-content">
          <XLeftPane
            task_description={this.props.task_description}
            full={true}
            frame_height={this.props.frame_height}
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

var component_list = {
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
}

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

var XContentLayout = getCorrectComponent('XContentLayout', null);
var XLeftPane = getCorrectComponent('XLeftPane', null);
var XRightPane = getCorrectComponent('XRightPane', null);
var XResponsePane = getCorrectComponent('XResponsePane', null);
var XTextResponse = getCorrectComponent('XTextResponse', null);
var XDoneResponse = getCorrectComponent('XDoneResponse', null);
var XIdleResponse = getCorrectComponent('XIdleResponse', null);
var XDoneButton = getCorrectComponent('XDoneButton', null);
var XChatPane = getCorrectComponent('XChatPane', null);
var XWaitingMessage = getCorrectComponent('XWaitingMessage', null);
var XMessageList = getCorrectComponent('XMessageList', null);
var XChatMessage = getCorrectComponent('XChatMessage', null);

function setComponentsForAgentID(agent_id) {
  XContentLayout = getCorrectComponent('XContentLayout', agent_id);
  XLeftPane = getCorrectComponent('XLeftPane', agent_id);
  XRightPane = getCorrectComponent('XRightPane', agent_id);
  XResponsePane = getCorrectComponent('XResponsePane', agent_id);
  XTextResponse = getCorrectComponent('XTextResponse', agent_id);
  XDoneResponse = getCorrectComponent('XDoneResponse', agent_id);
  XIdleResponse = getCorrectComponent('XIdleResponse', agent_id);
  XDoneButton = getCorrectComponent('XDoneButton', agent_id);
  XChatPane = getCorrectComponent('XChatPane', agent_id);
  XWaitingMessage = getCorrectComponent('XWaitingMessage', agent_id);
  XMessageList = getCorrectComponent('XMessageList', agent_id);
  XChatMessage = getCorrectComponent('XChatMessage', agent_id);
}

class MainApp extends React.Component {
  constructor(props) {
    super(props);
    let initialization_status = 'initializing';
    if (!doesSupportWebsockets()) {
      initialization_status = 'websockets_failure';
    }

    // TODO move constants to props rather than state
    this.state = {
      task_description: null,
      mturk_submit_url: null,
      frame_height: 650,
      socket_status: null,
      hit_id: HIT_ID, // gotten from template
      assignment_id: ASSIGNMENT_ID, // gotten from template
      worker_id: WORKER_ID, // gotten from template
      conversation_id: null,
      initialization_status: initialization_status,
      world_state: null,  // TODO cover onboarding and waiting separately
      is_cover_page: IS_COVER_PAGE, // gotten from template
      done_text: null,
      chat_state: 'idle',  // idle, text_input, inactive, done
      task_done: false,
      messages: [],
      agent_id: 'NewWorker',
      context: {}
    };
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevState.agent_id != this.state.agent_id) {
      setComponentsForAgentID(this.state.agent_id);
    }
  }

  handleIncomingHITData(data) {
    let task_description = data['task_description'];
    if (isMobile() && block_mobile) {
      task_description = (
        "<h1>Sorry, this task cannot be completed on mobile devices. " +
        "Please use a computer.</h1><br>Task Description follows:<br>" +
        data['task_description']
      );
    }

    this.setState({
      task_description: task_description,
      frame_height: data['frame_height'] || 650,
      mturk_submit_url: data['mturk_submit_url']
    });
  }

  componentDidMount() {
    getHitConfig((data) => this.handleIncomingHITData(data));
  }

  onMessageSend(text, data, callback) {
    if (text == '') {
      return;
    }
    this.socket_handler.handleQueueMessage(text, data, callback);
  }

  render() {
    let socket_handler = null;
    if (!this.state.is_cover_page) {
      socket_handler = <SocketHandler
        onMessageUpdate={() => this.setState({messages: this.state.messages})}
        onRequestMessage={() => this.setState({chat_state: 'text_input'})}
        onTaskDone={() => this.setState({
          task_done: true, chat_state: 'done', done_text: ''
        })}
        onInactiveDone={(inactive_text) => this.setState({
          task_done: true, chat_state: 'done', done_text: inactive_text
        })}
        onForceDone={allDoneCallback}
        onExpire={(expire_reason) => this.setState({
          chat_state: 'inactive', done_text: expire_reason
        })}
        onConversationChange={(world_state, conversation_id, agent_id) =>
          this.setState({
            world_state: world_state, conversation_id: conversation_id,
            agent_id: agent_id
          })
        }
        onSuccessfulSend={() => this.setState({
          chat_state: 'waiting', messages: this.state.messages
        })}
        onConfirmInit={() => this.setState({initialization_status: 'done'})}
        onFailInit={() => this.setState({initialization_status: 'failed'})}
        onStatusChange={(status) => this.setState({'socket_status': status})}
        assignment_id={this.state.assignment_id}
        conversation_id={this.state.conversation_id}
        worker_id={this.state.worker_id}
        agent_id={this.state.agent_id}
        hit_id={this.state.hit_id}
        initialization_status={this.state.initialization_status}
        messages={this.state.messages}
        task_done={this.state.task_done}
        ref={(m) => {this.socket_handler = m}}
      />;
    }
    return (
      <div>
        <BaseFrontend
          task_done={this.state.task_done}
          done_text={this.state.done_text}
          chat_state={this.state.chat_state}
          onMessageSend={(m, d, c) => this.onMessageSend(m, d, c)}
          socket_status={this.state.socket_status}
          messages={this.state.messages}
          agent_id={this.state.agent_id}
          task_description={this.state.task_description}
          initialization_status={this.state.initialization_status}
          is_cover_page={this.state.is_cover_page}
          frame_height={this.state.frame_height}
          context={this.state.context}
        />
        <MTurkSubmitForm
          assignment_id={this.state.assignment_id}
          hit_id={this.state.hit_id}
          worker_id={this.state.worker_id}
          mturk_submit_url={this.state.mturk_submit_url}/>
        {socket_handler}
      </div>
    );
  }
}

var main_app = <MainApp/>;

ReactDOM.render(main_app, document.getElementById('app'));
