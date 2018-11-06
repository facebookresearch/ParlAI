import React from 'react';
import ReactDOM from 'react-dom';
import {FormControl, Button} from 'react-bootstrap';
import CustomComponents from './components/custom.jsx'
import 'fetch';

/* ================= Data Model Constants ================= */

// Incoming message commands
const COMMAND_SEND_MESSAGE = 'COMMAND_SEND_MESSAGE'
const COMMAND_SHOW_DONE_BUTTON = 'COMMAND_SHOW_DONE_BUTTON'
const COMMAND_EXPIRE_HIT = 'COMMAND_EXPIRE_HIT'
const COMMAND_SUBMIT_HIT = 'COMMAND_SUBMIT_HIT'
const COMMAND_CHANGE_CONVERSATION = 'COMMAND_CHANGE_CONVERSATION'
const COMMAND_RESTORE_STATE = 'COMMAND_RESTORE_STATE'
const COMMAND_INACTIVE_HIT = 'COMMAND_INACTIVE_HIT'
const COMMAND_INACTIVE_DONE = 'COMMAND_INACTIVE_DONE'

// Socket function names
const SOCKET_OPEN_STRING = 'socket_open' // fires when a socket opens
const SOCKET_DISCONNECT_STRING = 'disconnect' // fires if a socket disconnects
const SOCKET_NEW_PACKET_STRING = 'new packet' // fires when packets arrive
const SOCKET_ROUTE_PACKET_STRING = 'route packet' // to send outgoing packets
const SOCKET_AGENT_ALIVE_STRING = 'agent alive' // to send alive packets

// Message types
const MESSAGE_TYPE_MESSAGE = 'MESSAGE'
const MESSAGE_TYPE_COMMAND = 'COMMAND'

// Packet types
const TYPE_ACK = 'ack';
const TYPE_MESSAGE = 'message';
const TYPE_HEARTBEAT = 'heartbeat';
const TYPE_PONG = 'pong';  // For messages back from the router on heartbeat
const TYPE_ALIVE = 'alive';

/* ================= Local Constants ================= */

const SEND_THREAD_REFRESH = 100;
const ACK_WAIT_TIME = 300; // Check for acknowledge every 0.3 seconds
const STATUS_ACK = 'ack';
const STATUS_INIT = 'init';
const STATUS_SENT = 'sent';
const CONNECTION_DEAD_MISSING_PONGS = 15;
const REFRESH_SOCKET_MISSING_PONGS = 5;
const HEARTBEAT_TIME = 2000;  // One heartbeat every 2 seconds

/* ============== Priority Queue Data Structure ============== */

// Initializing a 'priority queue'
function PriorityQueue() {
  this.data = []
}

// Pushes an element as far back as it needs to go in order to insert
PriorityQueue.prototype.push = function(element, priority) {
  priority = +priority
  for (var i = 0; i < this.data.length && this.data[i][0] < priority; i++);
  this.data.splice(i, 0, [element, priority])
}

// Remove and return the front element of the queue
PriorityQueue.prototype.pop = function() {
  return this.data.shift()
}

// Show the front element of the queue
PriorityQueue.prototype.peek = function() {
  return this.data[0]
}

// gets the size of the queue
PriorityQueue.prototype.size = function() {
  return this.data.length
}

/* ================= Utility functions ================= */

// log the data when the verbosity is greater than the level
// (where low level is high importance)
// Levels: 0 - Error, unusual behavior, something worth notifying
//         1 - Server interactions on the message level, commands
//         2 - Server interactions on the heartbeat level
//         3 - Potentially important function calls
//         4 - Practically anything
function log(data, level) {
  if (verbosity >= level) {
    console.log(data)
  }
}

// If we're in the amazon turk HIT page (within an iFrame) return True
function inMTurkHITPage() {
  try {
    return window.self !== window.top;
  } catch (e) {
    return true;
  }
}

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

// Checks to see if given conversation_id is for a waiting world
function isWaiting(conversation_id) {
  return (conversation_id != null && conversation_id.indexOf('w_') !== -1);
}

// Checks to see if given conversation_id is for a task world
function isTask(conversation_id) {
  return (conversation_id != null && conversation_id.indexOf('t_') !== -1);
}

// Checks to see if given conversation_id is for an onboarding world
function isOnboarding(conversation_id) {
  return (conversation_id != null && conversation_id.indexOf('o_') !== -1);
}

// Generate a random id
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Callback for submission
function allDoneCallback() {
  if (inMTurkHITPage()) {
    $("input#mturk_submit_button").click();
  }
}

class MTurkSubmitForm extends React.Component {
  /* Intentionally doesn't render anything, but prepares the form
  to submit data when the assignment is complete */
  render() {
    return (
      <form
        id="mturk_submit_form" action={this.props.mturk_submit_url}
        method="post" style={{"display": "none"}}>
          <input
            id="assignmentId" name="assignmentId"
            value={this.props.assignment_id} readOnly />
          <input id="hitId" name="hitId" value={this.props.hit_id} readOnly />
          <input
            id="workerId" name="workerId"
            value={this.props.worker_id} readOnly />
          <input
            type="submit" value="Submit"
            name="submitButton" id="mturk_submit_button" />
      </form>
    );
  }
}

class IdleResponse extends React.Component {
  render () {
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
          style={{'fontSize': '14pt', 'marginRight': '15px'}} />
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
    let done_button = <DoneButton />;
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
        this.state.textval, () => this.setState({textval: ''}));
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      e.stopPropagation();
      this.tryMessageSend();
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
        onKeyPress={() => this.handleKeyPress}
        onChange={(e) => this.setState({textval: e.target.value})}
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
    return messages.map(
      m => <ChatMessage
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
    if (this.props.messages.length != prevProps.messages.length) {
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
      wait_message = <WaitingMessage />;
    }

    return (
      <div id="right-top-pane" style={chat_style}>
        <MessageList {...this.props} />
        <ConnectionIndicator {...this.props} />
        {wait_message}
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
        <ChatPane {...this.props} />
        <ResponsePane {...this.props} />
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

class ResponsePane extends React.Component {
  render() {
    let response_pane = null;
    switch (this.props.chat_state) {
      case 'done':
      case 'inactive':
        response_pane = <DoneResponse
          {...this.props}
        />;
        break;
      case 'text_input':
      case 'waiting':
        response_pane = <TextResponse
          {...this.props}
          active={this.props.chat_state == 'text_input'}
        />;
        break;
      case 'idle':
      default:
        response_pane = <IdleResponse />;
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

class BaseFrontend extends React.Component {
  render () {
    let content = null;
    if (this.props.is_cover_page) {
      content = (
        <div className="row" id="ui-content">
          <LeftPane
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
      content = (
        <div className="row" id="ui-content">
          <LeftPane
            task_description={this.props.task_description}
            full={false}
            frame_height={this.props.frame_height}
          />
          <RightPane {...this.props} />
        </div>
      );
    }
    return (
      <div className="container-fluid" id="ui-container">
        {content}
      </div>
    );
  }
}

class MainApp extends React.Component {
  constructor(props) {
    super(props);
    this.q = new PriorityQueue();
    this.socket = null;
    let initialization_status = 'initializing';
    if (!doesSupportWebsockets()) {
      initialization_status = 'websockets_failure';
    }

    this.state = {
      task_description: null,
      mturk_submit_url: null,
      frame_height: 650,
      heartbeat_id: null,
      socket_closed: false,
      setting_socket: false,
      pongs_without_heartbeat: 0,
      socket_status: null,
      hit_id: HIT_ID, // gotten from template
      assignment_id: ASSIGNMENT_ID, // gotten from template
      worker_id: WORKER_ID, // gotten from template
      conversation_id: null,
      initialization_status: initialization_status,
      current_page: null,
      is_cover_page: IS_COVER_PAGE, // gotten from template
      done_text: null,
      chat_state: 'idle',  // idle, text_input, inactive, done
      task_done: false,
      messages: [],
      agent_id: 'NewWorker',
      packet_map: {},
      packet_callback: {},
      blocking_id: null,
      blocking_sent_time: null,
      blocking_intend_send_time: null,
      displayed_messages: []
    };
  }

  safePacketSend(packet) {
    console.log('sending');
    console.log(packet);
    if (this.socket.readyState == 0) {
      return;
    } else if (this.socket.readyState > 1) {
      log("Socket not in ready state, restarting if possible", 2);
      try {
        this.socket.close();
      } catch(e) {/* Socket already terminated */}
      this.setupSocket();
      return;
    }
    try {
      this.socket.send(JSON.stringify(packet));
    } catch(e) {
      log("Had error " + e + " sending message, trying to restart", 2);
      try {
        this.socket.close();
      } catch(e) {/* Socket already terminated */}
      this.setupSocket();
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

    console.log('TASK DATA');
    console.log(data);

    this.setState({
      task_description: task_description,
      frame_height: data['frame_height'] || 650,
      mturk_submit_url: data['mturk_submit_url']
    });
  }

  // Handles an incoming message
  handleNewMessage(new_message_id, message) {
    var agent_id = message.id;
    var message_text = message.text.replace(/(?:\r\n|\r|\n)/g, '<br />');
    if (this.state.displayed_messages.indexOf(new_message_id) !== -1) {
      // This message has already been seen and put up into the chat
      log(new_message_id + ' was a repeat message', 1);
      return;
    }

    log('New message, ' + new_message_id + ' from agent ' + agent_id, 1);
    this.state.displayed_messages.push(new_message_id);
    this.state.messages.push(message);
    this.setState({
      displayed_messages: this.state.displayed_messages,
      messages: this.state.messages
    })
  }

  parseSocketMessage(event) {
    let msg = JSON.parse(event.data)['content']
    if (msg.type === TYPE_HEARTBEAT) {
      // Heartbeats ensure we're not disconnected from the server
      log('received heartbeat: ' + msg.id, 4);
      this.setState({pongs_without_heartbeat: 0});
    } else if (msg.type === TYPE_PONG) {
      // Messages incoming from the router, ensuring that we're at least
      // connected to it.
      this.setState({
        pongs_without_heartbeat: this.state.pongs_without_heartbeat + 1,
        heartbeats_without_pong: 0
      });
      if (this.state.pongs_without_heartbeat >= 3) {
        this.setState({socket_status: 'reconnecting_server'});
      }
    } else if (msg.type === TYPE_ACK) {
      // Acks update messages so that we don't resend them
      log('received ack: ' + msg.id, 3);
      if (msg.id === this.state.blocking_id) {
        // execute ack callback if it exists
        if (this.state.packet_callback[msg.id]) {
          this.state.packet_callback[msg.id](this.state.packet_map[msg.id]);
        }
        this.setState({blocking_id: null, blocking_sent_time: null});
      }
      this.state.packet_map[msg.id].status = STATUS_ACK;
    } else if (this.state.blocking_id !== null) {
      return;  // shouldn't recieve messages while trying to send to world
    } else if (msg.type === TYPE_MESSAGE) {
      // Acknowledge the message, then act on it
      this.safePacketSend({
        type: SOCKET_ROUTE_PACKET_STRING,
        content: {
          id: msg.id,
          sender_id: msg.receiver_id,
          receiver_id: msg.sender_id,
          assignment_id: this.state.assignment_id,
          conversation_id: this.state.conversation_id,
          type: TYPE_ACK,
          data: null
        }
      });
      log(msg, 3);
      if (msg.data.type === MESSAGE_TYPE_COMMAND) {
        this.handleCommand(msg.data);
      } else if (msg.data.type === MESSAGE_TYPE_MESSAGE) {
        this.handleNewMessage(msg.id, msg.data);
      }
    }
  }

  // Handle incoming command messages
  handleCommand(msg) {
    let command = msg['text'];
    log('Recieved command ' + command, 1);
    if (command === COMMAND_SEND_MESSAGE) {
      // Update UI to wait for the worker to submit a message
      this.setState({chat_state: 'text_input'});
      log('Waiting for worker input', 4);
    } else if (command === COMMAND_SHOW_DONE_BUTTON) {
      // Update the UI to show the done button
      this.setState({task_done: true, chat_state: 'done', done_text: ''});
    } else if (command === COMMAND_INACTIVE_DONE) {
      // Update the UI to show done with additional inactive text
      this.closeSocket();
      // Call correct UI renderers
      this.setState({
        task_done: true, chat_state: 'done', done_text: msg['inactive_text']});
    } else if (command === COMMAND_SUBMIT_HIT) {
      // Force the hit to submit as done
      allDoneCallback();
    } else if (command === COMMAND_EXPIRE_HIT) {
      // Expire the hit unless it has already been marked as done
      if (!this.state.task_done) {
        this.setState({
          chat_state: 'inactive', done_text: msg['inactive_text']});
      }
    } else if (command === COMMAND_INACTIVE_HIT) {
      // Disable the hit, show the correct message
      this.setState({chat_state: 'inactive', done_text: msg['inactive_text']});
    } else if (command === COMMAND_RESTORE_STATE) {
      // Restore the messages from inside the data, call the command if needed
      let messages = msg['messages'];
      for (var i = 0; i < messages.length; i++) {
        this.handleNewMessage(messages[i]['message_id'], messages[i]);
      }

      last_command = msg['last_command'];
      if (last_command) {
        this.handleCommand(last_command);
      }
    } else if (command === COMMAND_CHANGE_CONVERSATION) {
      // change the conversation, refresh if needed
      log('current conversation_id: ' + conversation_id, 3);
      let conversation_id = msg['conversation_id'];
      log('new conversation_id: ' + conversation_id, 3);
      let agent_id = msg['agent_id'];
      let world_state = null;
      if (isWaiting(conversation_id)) {
        world_state = 'waiting';
      } else if (isOnboarding(conversation_id)) {
        world_state = 'onboarding';
      } else if (isTask(conversation_id)) {
        world_state = 'task';
      }
      this.setState({
        world_state: world_state, conversation_id: conversation_id,
        agent_id: agent_id
      })
      this.sendAlive();
    }
  }

  componentDidMount() {
    getHitConfig((data) => this.handleIncomingHITData(data));
    if (!this.state.is_cover_page) {
      this.setupSocket();
      window.setInterval(this.sendingThread.bind(this), SEND_THREAD_REFRESH);
    }
  }

  // way to send alive packets when expected to
  sendAlive() {
    this.sendPacket(
      TYPE_ALIVE,
      {
        hit_id: this.state.hit_id,
        assignment_id: this.state.assignment_id,
        worker_id: this.state.worker_id,
        conversation_id: this.state.conversation_id
      },
      true,
      true,
      () => this.setState({initialization_status: 'done'})
    );
  }

  closeSocket() {
    if (!this.state.socket_closed) {
      log("Socket closing", 3);
      this.socket.close();
      this.setState({socket_closed: true});
    } else {
      log("Socket already closed", 2);
    }
  }

  failInitialize() {
    if (this.state.initialization_status == 'initializing') {
      this.setState({initialization_status: 'failed'});
      this.closeSocket();
    }
  }

  // Sets up and registers the socket and the callbacks
  setupSocket() {
    if (this.state.setting_socket || this.state.socket_closed) {
      return;
    }

    // Note we are setting up the socket, ready to clear on failure
    this.setState({setting_socket: true});
    window.setTimeout(() => this.setState({setting_socket: false}), 4000);

    let url = window.location;
    if (url.hostname == 'localhost') {
      this.socket = new WebSocket('ws://' + url.hostname + ':' + url.port);
    } else {
      this.socket = new WebSocket('wss://' + url.hostname + ':' + url.port);
    }
    // TODO if socket setup fails here, see if 404 or timeout, then check
    // other reasonable domain. If that succeeds, assume the heroku server
    // died.

    this.socket.onmessage = this.parseSocketMessage.bind(this);

    this.socket.onopen = () => {
      log('Server connected.', 2);
      let setting_socket = false
      window.setTimeout(this.sendAlive.bind(this), 100);
      window.setTimeout(this.failInitialize.bind(this), 10000);
      let heartbeat_id = null;
      if (this.state.heartbeat_id == null) {
        heartbeat_id = window.setInterval(
          this.heartbeatThread.bind(this), HEARTBEAT_TIME);
      } else {
        heartbeat_id = this.state.heartbeat_id;
      }
      this.setState({
        heartbeat_id: heartbeat_id,
        setting_socket: setting_socket,
        socket_status: 'connected',
      });
    }

    this.socket.onerror = () => {
      log('Server disconnected.', 3);
      try {
        this.socket.close();
      } catch(e) {
        log('Server had error ' + e + ' when closing after an error', 1);
      }
      window.setTimeout(() => this.setupSocket(), 500);
    }

    this.socket.onclose = () => {
      log('Server closing.', 3);
      this.setState({socket_status: 'disconnected'});
    }
  }

  onMessageSend(text, callback) {
    if (text == '') {
      return;
    }

    let new_message_id = uuidv4();
    this.sendPacket(
      TYPE_MESSAGE,
      {
        text: text,
        id: this.state.agent_id,
        message_id: new_message_id,
        episode_done: false
      },
      true,
      true,
      (msg) => {
        this.state.messages.push(msg.data);
        this.setState({chat_state: 'waiting', messages: this.state.messages});
        callback();
      }
    );
  }

  sendHelper(msg, queue_time) {
    // Don't act on acknowledged packets
    if (msg.status !== STATUS_ACK) {
      // Find the event to send to
      let event_name = SOCKET_ROUTE_PACKET_STRING;
      if (msg.type === TYPE_ALIVE) {
        event_name = SOCKET_AGENT_ALIVE_STRING;
      }
      this.safePacketSend({type: event_name, content: msg});

      if (msg.require_ack) {
        if (msg.blocking) {
          // Block the thread
          this.setState({
            blocking_id: msg.id,
            blocking_sent_time: Date.now(),
            blocking_intend_send_time: queue_time
          });
        } else {
          // Check to see if the packet is acknowledged in the future
          this.q.push(msg, queue_time + ACK_WAIT_TIME);
        }
      }
    }
  }

  // Thread checks the message queue and handles pushing out new messages
  // as they are added to the queue
  sendingThread() {
    if (this.state.socket_closed) {
      return;
    }
    let blocking_id = this.state.blocking_id;

    // Can't act if something is currently blocking
    if (blocking_id === null) {
      // Can't act on an empty queue
      if (this.q.size() > 0) {
        // Can't act if the send time for the next thing to send
        // is in the future
        if (Date.now() > this.q.peek()[1]) {
          var item = this.q.pop();
          var msg = item[0];
          var t = item[1];
          this.sendHelper(msg, t);
        }
      }
    } else {
      let blocking_sent_time = this.state.blocking_sent_time;
      let packet_map = this.state.packet_map;
      let blocking_intend_send_time = this.state.blocking_intend_send_time;
      // blocking on packet `blocking_id`
      // See if we've waited too long for the packet to be acknowledged
      if ((Date.now() - blocking_sent_time) > ACK_WAIT_TIME) {
        log('Timeout: ' + blocking_id, 1);
        // Send the packet again now
        var m = packet_map[blocking_id];
        // Ensure that the send retries, we wouldn't be here if the ACK worked
        m.status = STATUS_SENT;
        this.sendHelper(packet_map[blocking_id], blocking_intend_send_time);
      }
    }
  }

  // Thread sends heartbeats through the socket for as long we are connected
  heartbeatThread() {
    console.log('heartbeat');
    console.log(this.state.pongs_without_heartbeat + ':' + this.state.heartbeats_without_pong);
    if (this.state.socket_closed) {
      // No reason to keep a heartbeat if the socket is closed
      window.clearInterval(this.state.heartbeat_id);
      this.setState({heartbeat_id: null});
      return;
    }

    if (this.state.pongs_without_heartbeat == REFRESH_SOCKET_MISSING_PONGS) {
      this.state.pongs_without_heartbeat += 1;
      try {
        console.log('too many pongs without heartbeat, refreshing');
        this.socket.close();
      } catch(e) {/* Socket already terminated */}
      window.clearInterval(this.state.heartbeat_id);
      this.setState({heartbeat_id: null});
      this.setupSocket();
    }

    // Check to see if we've disconnected from the server
    if (this.state.pongs_without_heartbeat > CONNECTION_DEAD_MISSING_PONGS) {
      this.closeSocket();
      let done_text = ('Our server appears to have gone down during the \
        duration of this HIT. Please send us a message if you\'ve done \
        substantial work and we can find out if the hit is complete enough to \
        compensate.');
      window.clearInterval(this.state.heartbeat_id);
      this.setState({
        heartbeat_id: null, chat_state: 'inactive', done_text: done_text
      });
    }

    var hb = {
      'id': uuidv4(),
      'receiver_id': '[World_' + TASK_GROUP_ID + ']',
      'assignment_id': this.state.assignment_id,
      'sender_id' : this.state.worker_id,
      'conversation_id': this.state.conversation_id,
      'type': TYPE_HEARTBEAT,
      'data': null
    };

    this.safePacketSend({type: SOCKET_ROUTE_PACKET_STRING, content: hb});

    this.setState({
      heartbeats_without_pong: this.state.heartbeats_without_pong + 1
    });
    if (this.state.heartbeats_without_pong >= 3) {
      this.setState({'socket_status': 'reconnecting_router'});
    } else if (this.state.heartbeats_without_pong >= 12) {
      this.closeSocket();
    }
  }

  // Enqueues a message for sending, registers the message and callback
  sendPacket(type, data, require_ack, blocking, callback) {
    var time = Date.now();
    var id = uuidv4();

    var msg = {
      id: id,
      type: type,
      sender_id: this.state.worker_id,
      assignment_id: this.state.assignment_id,
      conversation_id: this.state.conversation_id,  // TODO pull from something
      receiver_id: '[World_' + TASK_GROUP_ID + ']',
      data: data,
      status: STATUS_INIT,
      require_ack: require_ack,
      blocking: blocking
    };

    console.log('send');
    console.log(msg);

    this.q.push(msg, time);
    this.state.packet_map[id] = msg;
    this.state.packet_callback[id] = callback;
  }

  render() {
    return (
      <div>
        <BaseFrontend
          task_done={this.state.task_done}
          done_text={this.state.done_text}
          chat_state={this.state.chat_state}
          onMessageSend={(msg, callback) => this.onMessageSend(msg, callback)}
          socket_status={this.state.socket_status}
          messages={this.state.messages}
          agent_id={this.state.agent_id}
          task_description={this.state.task_description}
          initialization_status={this.state.initialization_status}
          is_cover_page={this.state.is_cover_page}
          frame_height={this.state.frame_height}
        />
        <MTurkSubmitForm
          assignment_id={this.state.assignment_id}
          hit_id={this.state.hit_id}
          worker_id={this.state.worker_id}
          mturk_submit_url={this.state.mturk_submit_url}/>
      </div>
    );
  }
}

if (CustomComponents.MainApp !== undefined) {
  MainApp = CustomComponents.MainApp;
}

var main_app = <MainApp/>;

ReactDOM.render(main_app, document.getElementById('app'));
