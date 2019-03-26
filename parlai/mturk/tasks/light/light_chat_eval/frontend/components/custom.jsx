/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';
import {getCorrectComponent} from './core_components.jsx';
import $ from 'jquery';

// Custom message component shows context if it exists:
class ChatMessage extends React.Component {
  render() {
    if (this.props.agent_id == 'persona' || this.props.agent_id == 'setting') {
      return null;  // Don't render state messages in the feed
    }
    let float_loc = 'left';
    let alert_class = 'alert-warning';
    if (this.props.is_self) {
      float_loc = 'right';
      alert_class = 'alert-info';
    }
    let action = null;
    let task_data = this.props.task_data;
    if (task_data !== undefined && !!task_data.action) {
      action = <span><br /><b>Action: </b><i>{task_data.action}</i></span>;
    }
    let duration = null;
    if (this.props.duration !== undefined) {
      let duration_seconds = Math.floor(this.props.duration / 1000) % 60;
      let duration_minutes = Math.floor(this.props.duration / 60000);
      let min_text = duration_minutes > 0 ? duration_minutes + ' min' : '';
      let sec_text = duration_seconds > 0 ? duration_seconds + ' sec' : '';
      duration = <small>
        <br /><i>Duration: </i>{min_text + ' ' + sec_text}
      </small>;
    }
    return (
      <div className={"row"} style={{'marginLeft': '0', 'marginRight': '0'}}>
        <div
          className={"alert " + alert_class} role="alert"
          style={{'float': float_loc, 'display': 'table'}}>
          <span style={{'fontSize': '16px'}}>
            <b>{this.props.agent_id}</b>: {this.props.message}
            {action}
            {duration}
          </span>
        </div>
      </div>
    );
  }
}

class ActionInput extends React.Component {
  constructor(props) {
    super(props);
    let actions = [];
    if (this.props.task_data.actions !== undefined) {
      actions = this.props.task_data.actions;
    } else {
      actions = ['loading'];
    }
    this.state = {
      'selectval': 'select one',
      'actions': actions,
      'sending': false,
    };
  }

  tryMessageSend() {
    if (this.state.selectval != 'select one' && this.props.active &&
        !this.state.sending) {
      this.setState({sending: true});
      this.props.onMessageSend(
        this.state.selectval,
        {},
        () => this.setState({selectval: 'select one', sending: false}));
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  getActionsDropdown() {
    let actions = null;
    if (this.props.task_data.actions !== undefined) {
      actions = this.props.task_data.actions;
    } else {
      actions = ['loading'];
    }
    actions = actions.concat(['select one'])
    let options = actions.map(
      (act_text) => <option key={act_text} value={act_text}>{act_text}</option>
    );
    let use_width = '80%';
    let gesture_selector = null;
    return (
      <div>
        <FormControl
          componentClass="select" placeholder="select"
          value={this.state.selectval}
          onChange={(e) => this.setState({selectval: e.target.value})}
          style={{float: 'left', width: use_width}}
          disabled={!this.props.active}>
          {options}
        </FormControl>
      </div>
    );
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
      height: "50px", width: "100%", display: "block",
    };
    let submit_style = {
      'width': '100px', 'height': '100%', 'fontSize': '16px',
      'marginLeft': '10px', 'padding': '0px'
    };

    let action_select = this.getActionsDropdown();

    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.selectval == 'select one' || !this.props.active ||
          this.state.sending}
        onClick={() => this.tryMessageSend()}>
          Send
      </Button>
    );

    let message = null;
    if (this.props.v_id == 'speech' || this.props.v_id == 'TestSpeech') {
      message = <ChatMessage
        is_self={true}
        agent_id={this.props.task_data.agent_id}
        message={this.state.selectval}
        task_data={this.props.task_data.curr_message_context}
        duration={undefined}/>;
    } else {
      message = <ChatMessage
        is_self={true}
        agent_id={this.props.task_data.agent_id}
        message={this.props.task_data.text}
        task_data={{action: this.state.selectval}}
        duration={undefined}/>;
    }

    let select_label = 'Selection:';
    if (this.props.v_id == 'emote' || this.props.v_id == 'TestEmote') {
      select_label = 'Gesture:'
    }

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}>
          <div style={input_style}>
            <b style={{float: 'left', 'paddingRight': '15px'}}>
              {select_label}
            </b>
            {action_select}
          </div>
          <div style={input_style}>
            {message}
          </div>
          <div style={input_style}>
            {submit_button}
          </div>
      </div>
    );
  }
}

class TestMessageList extends React.Component {
  makeMessages() {
    let agent_id = this.props.agent_id;
    let messages = [
      {
        id: 'Jester',
        text: 'Hey there guard, are you interested in hearing a joke?',
        task_data: {action: 'gesture wave'},
        message_id: 1,
      },
      {
        id: 'Guard',
        text: 'Sure, I have a moment to take a listen! And I love a good laugh',
        task_data: undefined,
        message_id: 2,
      },
      {
        id: 'Jester',
        text: 'Why couldn\'t the ogre make it to dinner after work?',
        task_data: undefined,
        message_id: 3,
      },
      {
        id: 'Guard',
        text: 'Hmmmm... I don\'t know, why?',
        task_data: {action: 'gesture ponder'},
        message_id: 4,
      },
      {
        id: 'Jester',
        text: 'Because they were absolutely SWAMPED!!!',
        task_data: undefined,
        message_id: 5,
      },
    ];

    if (this.props.task_data.turn === undefined ||
        this.props.task_data.turn == 'FIRST_TURN') {
      messages = messages.concat([{
        id: 'System',
        text: 'Please select the most appropriate gesture that makes sense ' +
              'in the next turn given the persona, setting, and ' +
              'conversation so far.',
        message_id: 6,
      }])
    } else {
      messages = messages.concat([
        {
          id: 'Guard',
          text: 'Bahahaha that\'s a great one! Where\'d you get that from?',
          task_data: {action: 'gesture laugh'},
          message_id: 6,
        },
        {
          id: 'Jester',
          text: 'I came up with it for the dumb king\'s most recent humoring ' +
                'but that fool doesn\'t take any jokes. Jeez I hate the guy...',
          task_data: {action: 'gesture groan'},
          message_id: 7,
        },
      ]);
      if (this.props.task_data.turn == 'SECOND_TURN') {
        messages = messages.concat([{
          id: 'System',
          text: 'Please select the most appropriate spoken response that ' +
                'makes the sense in the next turn given the persona, ' +
                'setting, and conversation so far.',
          message_id: 8,
        }])
      } else {
        messages = messages.concat([
          {
            id: 'Guard',
            text: 'Now you better watch your tongue Jester. I won\'t have ' +
                  'you badmouthing our king.',
            task_data: {action: 'gesture groan'},
            message_id: 8,
          },
          {
            id: 'Jester',
            text: 'Oh now what are you going to do to me? Our King is a ' +
                  'blithering idiot.',
            task_data: {action: 'gesture stare'},
            message_id: 9,
          },
        ]);
        if (this.props.task_data.turn == 'THIRD_TURN') {
          messages = messages.concat([{
            id: 'System',
            text: 'Please select the most appropriate action that makes the ' +
                  'sense in the next turn given the persona, setting, and ' +
                  'conversation so far.',
            message_id: 10,
          }])
        } else {
          messages = messages.concat([
            {
              id: 'Guard',
              text: 'You gotta get your senses straight. Hyah! ' +
                    'Consider this a warning...',
              task_data: {action: 'hit Jester'},
              message_id: 10,
            },
            {
              id: 'System',
              text: 'Thank you for taking the test! We will now give ' +
                    'you a real example.',
              message_id: 11,
            },
          ]);
        }
      }
    }

    console.log(this.props.task_data)
    if (this.props.task_data.wrong > 0) {
      messages = messages.concat([{
        id: 'System',
        text: 'Your previous entry was incorrect. You have made ' +
              this.props.task_data.wrong + ' mistakes so far.',
        message_id: 30,
      }])
    }

    if (this.props.task_data.turn == 'FAILED') {
      messages = [{
        id: 'System',
        'text': 'Sorry, you have failed to answer the questions within the ' +
        'allowed number of mistakes, and thus have failed the qualification ' +
        'test for this round. Please return the HIT.'
      }]
    }
    // Handles rendering messages from both the user and anyone else
    // on the thread - agent_ids for the sender of a message exist in
    // the m.id field.
    let XChatMessage = ChatMessage;
    let onClickMessage = this.props.onClickMessage;
    if (typeof onClickMessage !== 'function') {
      onClickMessage = (idx) => {};
    }
    return messages.map(
      (m, idx) =>
        <div key={m.message_id} onClick={() => onClickMessage(idx)}>
          <XChatMessage
            is_self={m.id == 'Guard'}
            agent_id={m.id}
            message={m.text}
            task_data={m.task_data}
            message_id={m.message_id}
            duration={this.props.is_review ? m.duration : undefined}/>
        </div>
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

class TaskMessageList extends React.Component {
  makeMessages() {
    let agent_id = this.props.agent_id;
    let messages = this.props.task_data.messages || [{'id': 'System', 'text': 'Loading'}];
    messages = [{
      'id': 'System',
      'text': 'Welcome to a new task, please look at your new context ' +
              'before starting.'
    }].concat(messages);

    if (agent_id == 'speech') {
      messages = messages.concat([{
        id: 'System',
        text: 'Please select the most appropriate spoken response given the ' +
              'persona, setting, given action/gesture, and conversation so far.',
        message_id: 6,
      }])
    } else if (agent_id == 'emote') {
      messages = messages.concat([{
        id: 'System',
        text: 'Please select the most appropriate gesture that matches the ' +
              'persona, setting, given spoken text, and conversation so far.',
        message_id: 6,
      }])
    } else if (agent_id == 'action') {
      messages = messages.concat([{
        id: 'System',
        text: 'Please select the most appropriate action that matches the ' +
              'persona, setting, given spoken text, and conversation so far.',
        message_id: 6,
      }])
    }

    // Handles rendering messages from both the user and anyone else
    // on the thread - agent_ids for the sender of a message exist in
    // the m.id field.
    let XChatMessage = ChatMessage;
    let onClickMessage = this.props.onClickMessage;
    if (typeof onClickMessage !== 'function') {
      onClickMessage = (idx) => {};
    }
    return messages.map(
      (m, idx) =>
        <div key={'message-'+ idx} onClick={() => onClickMessage(idx)}>
          <XChatMessage
            is_self={m.id == this.props.task_data.agent_id}
            agent_id={m.id}
            message={m.text}
            task_data={m.task_data}
            message_id={m.message_id}
            duration={this.props.is_review ? m.duration : undefined}/>
        </div>
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

class PersonaSettingContext extends React.Component {
  render () {
    let persona_context = 'Loading...';
    let character_name = '';
    let setting_context = 'Loading...';
    let setting_name = '';
    let partner_name = 'Loading...';
    if (this.props.task_data.persona !== undefined) {
      persona_context = this.props.task_data.persona;
      character_name = this.props.task_data.base_name;
      partner_name = this.props.task_data.partner_name;
    }
    if (this.props.task_data.setting !== undefined) {
      setting_context = this.props.task_data.setting;
      setting_name = this.props.task_data.setting_name;
    }
    let context = (
      <div>
        <h1></h1>
        <h2>Persona: {character_name} </h2>
        <p>
          {persona_context}
        </p>
        <h2>Setting: {setting_name} </h2>
        <p>
          {setting_context}
        </p>
        <h2>Chat Partner: {partner_name} </h2>
      </div>
    )
    return (
      <div>
          <h1>
            Your Context:
            <span style={{color: 'red'}}>(READ BEFORE STARTING)</span>
          </h1>
          <hr style={{'borderTop': '1px solid #555'}} />
          {context}
      </div>
    );
  }
}

class TaskDescription extends React.Component {
  render () {
    let header_text = "Role Playing in Multiplayer Text Adventure";
    let content = (
      <div>
        <h3>Instructions</h3>
        <h4>You will be playing a character in a text adventure game</h4>
        <p>
          For this task, you will be selecting one of a few multiple choice
          options for the most likely thing to say, gesture, or act in a
          text adventure game conversation. You should select what you believe
          to be the most likely correct option given the context and the
          parts of the message that you have. One of the options was provided
          by a worker in a previous task, and for the instance of this task
          we consider that option to be the correct one.
          <br />
          Your first time taking this task you will be paired with a test
          version. Passing this task will allow you to continue to work on the
          task moving forward. Failing this test version will prevent you from
          being able to work on the task, and you'll need to return it.
          <b>You have 3 chances for incorrect answers.</b>
          <br />
          The top 10% of workers (workers who's selections agree with the
          original inputs) will get a 10% bonus across all tasks done.
          <br />
          Workers who are flagged for picking randomly may be blocked and have
          their HITs rejected. It is very possible to do better than guessing
          randomly, and we will be able to tell if you are doing so.
        </p>
      </div>
    );
    return (
      <div>
          <h1>{header_text}</h1>
          <hr style={{'borderTop': '1px solid #555'}} />
          {content}
      </div>
    );
  }
}

export default {
  XTextResponse: {
    'default': ActionInput,
  },
  XMessageList: {
    'TestEmote': TestMessageList,
    'TestSpeech': TestMessageList,
    'TestAct': TestMessageList,
    'default': TaskMessageList,
  },
  XTaskDescription: {'default': TaskDescription},
  XChatMessage: {'default': ChatMessage},
  XContextView: {'default': PersonaSettingContext},
};
