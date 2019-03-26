/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';
import {getCorrectComponent} from './core_components.jsx';
import $ from 'jquery';


class GraphWorldInput extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      'textval': '',
      'selectval': '',
      'gestureval': 'applaud',
      'sending': false,
    };
  }

  getActions() {
    let actions = [];
    if (this.props.task_data.actions !== undefined) {
      actions = this.props.task_data.actions;
    }
    return actions;
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $("input#id_text_input").focus();
    }
  }

  tryMessageSend() {
    if (this.state.textval != '' && this.props.active && !this.state.sending) {
      this.setState({sending: true});
      let use_action = this.state.selectval;
      if (use_action == 'gesture') {
        use_action += ' ' + this.state.gestureval;
      }
      this.props.onMessageSend(
        this.state.textval,
        {
          'action': use_action,
        },
        () => this.setState({textval: '', selectval: '', sending: false}));
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSend();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  getGestureOptions() {
    let GESTURES = [
      'applaud', 'blush', 'cry', 'dance',
      'frown', 'gasp', 'grin', 'groan', 'growl',
      'yawn', 'laugh', 'nod', 'nudge', 'ponder', 'pout', 'scream',
      'shrug', 'sigh', 'smile', 'stare', 'wave', 'wink',
    ];
    return GESTURES.map(
      (g) => <option key={g + '-action'} value={g}>{g}</option>
    )
  }

  getActionsDropdown() {
    let actions = this.getActions();
    let options = actions.map(
      (act_text) => <option key={act_text} value={act_text}>{act_text}</option>
    );
    let use_width = '80%';
    let gesture_selector = null;
    if (this.state.selectval == 'gesture') {
      use_width = '40%';
      gesture_selector = (
        <FormControl
          componentClass="select" placeholder="select"
          value={this.state.gestureval}
          onChange={(e) => this.setState({gestureval: e.target.value})}
          style={{float: 'left', width: use_width}}
          disabled={!this.props.active}>
          {this.getGestureOptions()}
        </FormControl>
      )
    }
    return (
      <div>
        <FormControl
          componentClass="select" placeholder="select"
          value={this.state.selectval}
          onChange={(e) => this.setState({selectval: e.target.value})}
          style={{float: 'left', width: use_width}}
          disabled={!this.props.active}>
          <option key='nothing-action' value=''>Speak only</option>
          <option key='gesture-action' value='gesture'>gesture</option>
          {options}
        </FormControl>
        {gesture_selector}
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
        placeholder="Enter spoken text here"
        onKeyPress={(e) => this.handleKeyPress(e)}
        onChange={(e) => this.setState({textval: e.target.value})}
        disabled={!this.props.active || this.state.sending}/>
    );

    let action_select = this.getActionsDropdown();

    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.textval == '' || !this.props.active || this.state.sending}
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
            <b style={{float: 'left', 'paddingRight': '15px'}}>
              Action: <br/> (optional)
            </b>
            {action_select}
          </div>
          <div style={input_style}>
            {text_input}
            {submit_button}
          </div>
      </div>
    );
  }
}

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

class PersonaSettingContext extends React.Component {
  render () {
    let persona_context = 'Loading...';
    let character_name = '';
    let setting_context = 'Loading...';
    if (this.props.task_data.persona !== undefined) {
      persona_context = this.props.task_data.persona;
      character_name = this.props.task_data.base_name;
    }
    if (this.props.task_data.setting !== undefined) {
      setting_context = this.props.task_data.setting;
    }
    let context = (
      <div>
        <h1></h1>
        <h2>Persona: {character_name}</h2>
        <h4>(pretend to be this, talk a bit about yourself)</h4>
        <p>
          {persona_context}
        </p>
        <h2>Setting: </h2>
        <h4>(pretend to be here, use it in conversation sometimes)</h4>
        <p>
          {setting_context}
        </p>
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
    let context = null;
    if (this.props.is_cover_page) {
      // Return a view for the task instructions
      let messages = [
        {
          message_id: 1,
          id: 'Traveler',
          text: 'ah finally I can get myself a good drink!',
          task_data: {action: 'get beer'},
        },
        {
          message_id: 2,
          id: 'Bartender',
          text: 'Woah there buddy, are you going to pay for that?',
          task_data: {action: ''},
        },
        {
          message_id: 3,
          id: 'Traveler',
          text: 'Why does it matter? I need this drink!',
          task_data: {action: 'drink beer'},
        },
        {
          message_id: 4,
          id: 'Bartender',
          text: 'Sir I cannot just have patrons drinking my beer for free.',
          task_data: {action: ''},
        },
        {
          message_id: 5,
          id: 'Traveler',
          text: "Look I have nothing left, can't you cut me a break?",
          task_data: {action: ''},
        },
        {
          message_id: 6,
          id: 'Bartender',
          text: "Nothing comes for free at my tavern. Maybe you can help me clean up?",
          task_data: {action: 'Gives you a dish towel'},
        },
        {
          message_id: 7,
          id: 'Traveler',
          text: "You think I'm going to clean just because you asked? You " +
                "sound like my ex wife.",
          task_data: {action: ''},
        },
        {
          message_id: 8,
          id: 'Bartender',
          text: "Let me lend you an ear and we'll go from there - what " +
                "brings you to my bar?",
          task_data: {action: ''},
        },
        {
          message_id: 9,
          id: 'Traveler',
          text: "My wife left me because we never seemed to agree. She took " +
                "all I had and turned my town against me. I'm looking for " +
                "a new life.",
          task_data: {action: ''},
        },
        {
          message_id: 10,
          id: 'Bartender',
          text: "Ah you come with a tough tale - how about you give me that " +
                "towel back and I let you have that first one on the house?",
          task_data: {action: ''},
        },
        {
          message_id: 11,
          id: 'Traveler',
          text: "Seems like a fine deal to me. If you need a hand around " +
                "here long term I could be interested. Otherwise it's off " +
                "to the next town for me.",
          task_data: {action: 'Give dish towel to bartender'},
        },
      ];
      let XMessageList = getCorrectComponent('XMessageList', null);
      context = (
        <div>
          Below are the first few turns of an example chat.
          <h4>Example:</h4>
          <p>
            <b>Persona: Traveler</b><br />
            I am a weary traveler. I haven't had anything to drink in days
            and am on the verge of collapsing. I left my old life behind after
            my wife left me. She hated how disagreeable I could be.
          </p>
          <p>
            <b>Setting: </b><br />
            You are in a dusty tavern. There's a bar up front, but it seems
            that there aren't many patrons. The walls are peeling and it smells
            as if a flood passed through and was poorly cleaned. There's a
            beer here. There's a bartender here.
          </p>
          <XMessageList agent_id={'Traveler'} messages={messages}/>
          <br/>
          <h4>Please accept the task if you're ready</h4>
        </div>
      )
    }
    let content = (
      <div>
        <h3>Instructions</h3>
        <h4>You will be playing a character in a text adventure game</h4>
        <p>
          For this task you'll be paired with another worker who is also
          playing a character in the game. You will take turns speaking and
          taking game actions, such as getting and giving items. <b> Talk to
          them, learn about their persona, have a conversation about the
          given location. </b>
          <br />
          <b>You SHOULD NOT act every turn</b>, you should only
          take an action when it makes sense with what you say. Do not simply
          copy your persona into the chat, but be sure to abide by the persona.
          <br />
          Please use the given character and context in your conversation as
          it makes sense to. <b>If your persona doesn't at all match how
          you talk through the conversation, the hit may be rejected. </b>
          <br />
          Some character types (such as animals) may not say very much,
          however we expect you to try your best to play the character given.
          Some characters are given as plural (such as wise men), assume you
          are the singular (a wise man).
          <br />
          <b>OFFENSIVE LANGUAGE WILL LEAD TO A REJECTION</b><br />
          Please be mindful of your fellow worker and try to answer promptly.
          Workers who take unreasonably long times to respond will be soft
          blocked from these tasks.
        </p>
        {context}
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
  XTextResponse: {'default': GraphWorldInput},
  XTaskDescription: {'default': TaskDescription},
  XChatMessage: {'default': ChatMessage},
  XContextView: {'default': PersonaSettingContext},
};
