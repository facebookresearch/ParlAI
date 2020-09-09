/*
* Copyright (c) Facebook, Inc. and its affiliates.
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';
import $ from 'jquery';

let COVER_PAGE = (
  <div>
    <h2><b>PLEASE READ THE FOLLOWING CAREFULLY:</b></h2>
    <p style={{fontSize:'125%'}}>
      <b>
        <span style={{color: 'red'}}><i>WARNING</i></span>: You may see and/or
        be asked to generate offensive content.
      </b>
      <br></br><br></br>
      You will be asked to send messages that are either
      <span style={{color: 'green'}}><b> OK</b></span> or
      <span style={{color: 'red'}}><b> NOT OK</b></span> to send in
      a friendly conversation.
    <br></br>
    <b>For each turn, you will be asked to come up with a message either by:</b>
    </p>
      <ol style={{fontSize: '115%'}}>
      <li>Generating one <span style={{color: 'purple'}}><b>from scratch</b></span>.</li>
      <li>Generating one based <span style={{color: 'purple'}}><b>on a given topic</b></span>.
        Pretend as though you are discussing that topic in a conversation.</li>
    </ol>
    <br></br>
    <p style={{fontSize: '125%'}}>
      If you are ready, please click
      <span style={{color: 'blue'}}><b> Accept HIT</b></span> to
      start this task.
    </p>
  </div>
)

let TOPIC = (
  <div>
    <h2><b>PLEASE READ THE FOLLOWING CAREFULLY:</b></h2>
    <h3 style={{color: 'blue'}}><b>Generate Messages on a Topic:</b></h3>
    <p style={{fontSize:'125%'}}>
      For each message, you will be given a topic.
       You will be asked to send a message that is either
      <span style={{color: 'green'}}><b> OK</b></span> or
      <span style={{color: 'red'}}><b> NOT OK</b></span> to send in a friendly
      conversation about that given topic.

      <br></br><br></br>
      <b>
      <i>WARNING:</i> THE EXAMPLES BELOW MAY CONTAIN OFFENSIVE CONTENT:
      </b>
    </p>
    <h3><b> Some examples: </b></h3>
    <ul style={{fontSize: '125%'}}>
      <li>Topic: <b>EuroBasket</b></li>
      <li><span style={{color: 'red'}}><b>NOT OK</b></span>: Europeans can't play basketball!! Stick to soccer.</li>
      <li><span style={{color: 'green'}}><b>OK</b></span>: What position did you used to play? I am more of a soccer fan, myself.</li>
    </ul>
    <br></br>
    <ul style={{fontSize: '125%'}}>
      <li>Topic: <b>Sephora</b></li>
      <li><span style={{color: 'red'}}><b>NOT OK</b></span>: Every woman who wears makeup is a stuck up loser.</li>
      <li><span style={{color: 'green'}}><b>OK</b></span>: I've never been to Sephora, I don't wear much makeup to be honest.</li>
    </ul>
  </div>
)

let SCRATCH = (
  <div>
    <h2><b>PLEASE READ THE FOLLOWING CAREFULLY:</b></h2>
    <h3 style={{color: 'blue'}}><b>Generate Messages from Scratch:</b></h3>
    <p style={{fontSize:'125%'}}>
      You will be asked to send a message that is either
      <span style={{color: 'green'}}><b> OK</b></span> or
      <span style={{color: 'red'}}><b> NOT OK</b></span> to send in a friendly
      conversation.

      <br></br><br></br>
      <b>
      <i>WARNING:</i> THE EXAMPLES BELOW MAY CONTAIN OFFENSIVE CONTENT:
      </b>
    </p>
    <h3><b> Some examples: </b></h3>
    <ul style={{fontSize: '125%'}}>
      <li><span style={{color: 'red'}}><b>NOT OK</b></span>: Wtf dude, people who care about politics are all morons.</li>
      <li><span style={{color: 'green'}}><b>OK</b></span>: What do you do for work? I work in construction.</li>
      <li><span style={{color: 'red'}}><b>NOT OK</b></span>: Sounds like you have daddy issues...</li>
      <li><span style={{color: 'green'}}><b>OK</b></span>: Yeah, I love pizza too. What kind of toppings do you like?</li>
    </ul>
  </div>
);

let ONBOARDING = (
  <div style={{fontSize: '125%'}}>
    For each message you will be asked the following: <br></br>
    <b><i><span style={{color: 'blue'}}>Is this message OK for a person you just met
    online to send at some stage in a friendly conversation?</span></i></b>
    <br></br><br></br>

    <span style={{color: '#540773'}}>
    The message you see may be from the <b>beginning, middle, or end </b>
    of a conversation, but the rest of the conversation will <b>not</b> be shown.
    <br></br>
    <b>ASSUME THE MESSAGE MAKES SENSE WITHIN THE CONTEXT OF THE CONVERSATION!</b>
    </span>
    <br></br><br></br>
    <u>You will have <b>three choices</b> for each message:</u><br></br>
    <span style={{color: 'green'}}><b>OK</b></span>: would be fine
    <br></br>
    <span style={{color: '#daa520'}}><b>Maybe OK</b></span>: might be OK not sure
    <br></br>
    <span style={{color: 'red'}}><b>Not OK</b></span>: would generally not be OK
    <br></br><br></br>

    <b>NOTE:</b><i> Please <b>disgregard</b> any spelling or grammatical errors when making your decision!</i>

    <br></br><br></br>
    You may be asked to provide a reason for your response.
    <br></br><br></br>

    <b>Some examples of <span style={{color: 'green'}}>OK</span> messages that we found some workers were confused about</b>:
    <ul>
      <li>interesting. i am a website designer. pretty much spend all my time on the computer.</li>
      <b>Reason it's OK:</b> <i>talking about your job is OK.</i> <br></br><br></br>
      <li>okay but i was published in new yorker once.</li>
      <b>Reason it's OK:</b> <i>talking about your accomplishments is OK.</i> <br></br><br></br>
      <li>i work as a gourmet cook who also has a pitch perfect voice.</li>
      <b>Reason it's OK:</b> <i>talking about your job is OK, grammar is bad but that's OK.</i> <br></br><br></br>
      <li>yes, i believe so.</li>
      <b>Reason it's OK:</b> <i>in most contexts for which this is a natural reply it would be OK.</i> <br></br><br></br>
    </ul>
    <br></br>
  </div>
)

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

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $("input#id_text_input").focus();
    }
  }

  tryMessageSendText() {
    if (this.state.textval != '' && this.props.active) {
      this.props.onMessageSend(
        this.state.textval, {}, () => this.setState({textval: ''}));
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      this.tryMessageSendText();
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  render() {
  // BUTTON RESPONSE
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
        'width': '200px', 'height': '100%', 'fontSize': '16px',
        'float': 'left', 'marginLeft': '10px', 'padding': '0px'
      };

      let reject_button = (
        <Button
          className="btn btn-danger"
          style={submit_style}
          id="id_reject_chat_button"
          disabled={!this.props.active}
          onClick={() => this.tryMessageSend('not ok')}>
            Not OK
        </Button>
      );

      let approve_button = (
        <Button
          className="btn btn-success"
          style={submit_style}
          id="id_approve_chat_button"
          disabled={!this.props.active}
          onClick={() => this.tryMessageSend('ok')}>
            OK
        </Button>
      );

      let maybe_button = (
        <Button
          className="btn btn-warning"
          style={submit_style}
          id="id_maybe_chat_button"
          disabled={!this.props.active}
          onClick={() => this.tryMessageSend('maybe ok')}>
            Maybe OK
        </Button>
      );

      return (
        <div
          id="response-type-text-input"
          className="response-type-module"
          style={pane_style}>
            <div style={input_style}>
              {approve_button}
              {maybe_button}
              {reject_button}
            </div>
        </div>
      );

  }
}

class HTMLChatMessage extends React.Component {
  render() {
    let float_loc = 'left';
    let alert_class = 'alert-warning';
    if (this.props.is_self) {
      float_loc = 'right';
      alert_class = 'alert-info';
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
            <b>{this.props.agent_id}</b>:&nbsp;
            <span dangerouslySetInnerHTML={{__html: this.props.message}}/>
          </span>
          {duration}
        </div>
      </div>
    );
  }
}

class TaskDescription extends React.Component {
  render () {
    let header_text = "Write conversational messages!"
    let context = null;
    // Return a view for the task instructions
    if (this.props.is_cover_page) {
      context = COVER_PAGE
    }
    else if (this.props.agent_id == 'Topic') {
      context = TOPIC
    }
    else if (this.props.agent_id == 'Scratch') {
      context = SCRATCH
    }
    else {
      context = ONBOARDING
    }

    return (
      <div>
          <h1><b>{header_text}</b></h1>
          {context}
      </div>
    );
  }
}

// Package components
var ChatMessageHolder = {
  'Scratch': HTMLChatMessage,
  'Topic': HTMLChatMessage,
  'Modify': HTMLChatMessage,
  'Onboarding Evaluator': HTMLChatMessage,
}

var TextResponseHolder = {
  'Onboarding Evaluator': EvaluationResponse,
}

export default {
  // ComponentName: CustomReplacementComponentMap
  XChatMessage: ChatMessageHolder,
  XTextResponse: TextResponseHolder,
  XTaskDescription: {'default': TaskDescription},
};
