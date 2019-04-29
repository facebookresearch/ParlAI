/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import ReactDOM from 'react-dom';
import {
  FormControl,
  Button,
  ButtonGroup,
  InputGroup,
  FormGroup,
  MenuItem,
  DropdownButton,
  Badge,
  Popover,
  Overlay,
  Nav,
  NavItem,
  Col,
  ControlLabel,
  Form,
} from 'react-bootstrap';
import Slider from 'rc-slider';
import $ from 'jquery';

import 'rc-slider/assets/index.css';

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
    let duration = null;
    if (this.props.duration !== undefined) {
      let duration_seconds = Math.floor(this.props.duration / 1000) % 60;
      let duration_minutes = Math.floor(this.props.duration / 60000);
      let min_text = duration_minutes > 0 ? duration_minutes + ' min' : '';
      let sec_text = duration_seconds > 0 ? duration_seconds + ' sec' : '';
      duration = (
        <small>
          <br />
          <i>Duration: </i>
          {min_text + ' ' + sec_text}
        </small>
      );
    }
    return (
      <div className={'row'} style={{ marginLeft: '0', marginRight: '0' }}>
        <div
          className={'alert ' + alert_class}
          role="alert"
          style={{ float: float_loc, display: 'table' }}
        >
          <span style={{ fontSize: '16px', whiteSpace: 'pre-wrap' }}>
            <b>{this.props.agent_id}</b>: {this.props.message}
          </span>
          {duration}
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
    let onClickMessage = this.props.onClickMessage;
    if (typeof onClickMessage !== 'function') {
      onClickMessage = idx => {};
    }
    return messages.map((m, idx) => (
      <div key={m.message_id} onClick={() => onClickMessage(idx)}>
        <XChatMessage
          is_self={m.id == agent_id}
          agent_id={m.id}
          message={m.text}
          task_data={m.task_data}
          message_id={m.message_id}
          duration={this.props.is_review ? m.duration : undefined}
        />
      </div>
    ));
  }

  render() {
    return (
      <div id="message_thread" style={{ width: '100%' }}>
        {this.makeMessages()}
      </div>
    );
  }
}

class ConnectionIndicator extends React.Component {
  render() {
    let indicator_style = {
      opacity: '1',
      fontSize: '11px',
      color: 'white',
      float: 'right',
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
        disabled={true}
      >
        {text}
      </button>
    );
  }
}

class VolumeControl extends React.Component {
  constructor(props) {
    super(props);
    this.state = { slider_shown: false };
  }

  render() {
    let volume_control_style = {
      opacity: '1',
      fontSize: '11px',
      color: 'white',
      float: 'right',
      marginRight: '10px',
    };

    let slider_style = {
      height: 26,
      width: 150,
      marginRight: 14,
      float: 'left',
    };

    let content = null;
    if (this.state.slider_shown) {
      content = (
        <div style={volume_control_style}>
          <div style={slider_style}>
            <Slider
              onChange={v => this.props.onVolumeChange(v / 100)}
              style={{ marginTop: 10 }}
              defaultValue={this.props.volume * 100}
            />
          </div>
          <Button onClick={() => this.setState({ slider_shown: false })}>
            <span
              style={{ marginRight: 5 }}
              className="glyphicon glyphicon-remove"
            />
            Hide Volume
          </Button>
        </div>
      );
    } else {
      content = (
        <div style={volume_control_style}>
          <Button onClick={() => this.setState({ slider_shown: true })}>
            <span
              className="glyphicon glyphicon glyphicon-volume-up"
              style={{ marginRight: 5 }}
              aria-hidden="true"
            />
            Volume
          </Button>
        </div>
      );
    }
    return content;
  }
}

class ChatBox extends React.Component {
  state = {
    hidden: true,
    msg: '',
  };

  smoothlyAnimateToBottom() {
    if (this.bottomAnchorRef) {
      this.bottomAnchorRef.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }
  }

  instantlyJumpToBottom() {
    if (this.chatContainerRef) {
      this.chatContainerRef.scrollTop = this.chatContainerRef.scrollHeight;
    }
  }

  componentDidMount() {
    this.instantlyJumpToBottom();
  }

  componentDidUpdate(prevProps, prevState) {
    // Use requestAnimationFrame to defer UI-based updates
    // until the next browser paint
    if (prevState.hidden === true && this.state.hidden === false) {
      requestAnimationFrame(() => {
        this.instantlyJumpToBottom();
      });
    } else if (prevProps.off_chat_messages !== this.props.off_chat_messages) {
      requestAnimationFrame(() => {
        this.smoothlyAnimateToBottom();
      });
    }
  }

  // TODO: Replace with enhanced logic to determine if the
  // chat message belongs to the current user.
  isOwnMessage = message => message.owner === 0;

  render() {
    const unreadCount = this.props.has_new_message;
    const messages = this.props.off_chat_messages || [];

    return (
      <div style={{ float: 'right', marginRight: 7 }}>
        <Button
          onClick={() => this.setState({ hidden: !this.state.hidden })}
          ref={el => {
            this.buttonRef = el;
          }}
        >
          Chat Messages&nbsp;
          {!!unreadCount && (
            <Badge style={{ backgroundColor: '#d9534f', marginLeft: 3 }}>
              {unreadCount}
            </Badge>
          )}
        </Button>

        <Overlay
          rootClose
          show={!this.state.hidden}
          onHide={() => this.setState({ hidden: true })}
          placement="bottom"
          target={this.buttonRef}
        >
          <Popover id="chat_messages" title={'Chat Messages'}>
            <div
              className="chat-list"
              ref={el => {
                this.chatContainerRef = el;
              }}
              style={{ minHeight: 300, maxHeight: 300, overflowY: 'scroll' }}
            >
              {messages.map((message, idx) => (
                <div
                  key={idx}
                  style={{
                    textAlign: this.isOwnMessage(message) ? 'right' : 'left',
                  }}
                >
                  <div
                    style={{
                      borderRadius: 4,
                      marginBottom: 10,
                      padding: '5px 10px',
                      display: 'inline-block',
                      ...(this.isOwnMessage(message)
                        ? {
                            marginLeft: 20,
                            textAlign: 'right',
                            backgroundColor: '#dff1d7',
                          }
                        : {
                            marginRight: 20,
                            backgroundColor: '#eee',
                          }),
                    }}
                  >
                    {message.msg}
                  </div>
                </div>
              ))}
              <div
                className="bottom-anchor"
                ref={el => {
                  this.bottomAnchorRef = el;
                }}
              />
            </div>
            <form
              style={{ paddingTop: 10 }}
              onSubmit={e => {
                e.preventDefault();
                if (this.state.msg === '') return;
                this.props.onMessageSend(this.state.msg);
                this.setState({ msg: '' });
              }}
            >
              <FormGroup>
                <InputGroup>
                  <FormControl
                    type="text"
                    value={this.state.msg}
                    onChange={e => this.setState({ msg: e.target.value })}
                  />
                  <InputGroup.Button>
                    <Button
                      className="btn-primary"
                      disabled={this.state.msg === ''}
                      type="submit"
                    >
                      Send
                    </Button>
                  </InputGroup.Button>
                </InputGroup>
              </FormGroup>
            </form>
          </Popover>
        </Overlay>
      </div>
    );
  }
}

class ChatNavbar extends React.Component {
  state = {
    // TODO: replace hardcoded initial chat state with some API integration
    chat: [{ msg: 'hey', owner: 3 }, { msg: 'anyone else there?', owner: 3 }],
  };

  render() {
    // const displayChatBox = true;
    const displayChatBox = this.props.displayChatBox || false;
    let nav_style = {
      position: 'absolute',
      backgroundColor: '#EEEEEE',
      borderColor: '#e7e7e7',
      height: 46,
      top: 0,
      borderWidth: '0 0 1px',
      borderRadius: 0,
      right: 0,
      left: 0,
      zIndez: 1030,
      padding: 5,
    };
    return (
      <div style={nav_style}>
        <ConnectionIndicator {...this.props} />
        <VolumeControl {...this.props} />
        {displayChatBox && (
          <ChatBox
            off_chat_messages={this.state.chat}
            onMessageSend={msg =>
              this.setState({ chat: [...this.state.chat, { msg, owner: 0 }] })
            }
            has_new_message={2}
          />
        )}
      </div>
    );
  }
}

class Hourglass extends React.Component {
  render() {
    // TODO move to CSS document
    let hourglass_style = {
      marginTop: '-1px',
      marginRight: '5px',
      display: 'inline',
      float: 'left',
    };

    // TODO animate?
    return (
      <div id="hourglass" style={hourglass_style}>
        <span className="glyphicon glyphicon-hourglass" aria-hidden="true" />
      </div>
    );
  }
}

class WaitingMessage extends React.Component {
  render() {
    let message_style = {
      float: 'left',
      display: 'table',
      backgroundColor: '#fff',
    };
    let text = 'Waiting for the next person to speak...';
    if (this.props.world_state == 'waiting') {
      text = 'Waiting to pair with a task...';
    }
    return (
      <div
        id="waiting-for-message"
        className="row"
        style={{ marginLeft: '0', marginRight: '0' }}
      >
        <div className="alert alert-warning" role="alert" style={message_style}>
          <Hourglass />
          <span style={{ fontSize: '16px' }}>{text}</span>
        </div>
      </div>
    );
  }
}

class ChatPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = { chat_height: this.getChatHeight() };
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.message_count != prevProps.message_count) {
      $('div#message-pane-segment').animate(
        {
          scrollTop: $('div#message-pane-segment').get(0).scrollHeight,
        },
        500
      );
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

  handleResize() {
    if (this.getChatHeight() != this.state.chat_height) {
      this.setState({ chat_height: this.getChatHeight() });
    }
  }

  render() {
    let v_id = this.props.v_id;
    let XMessageList = getCorrectComponent('XMessageList', v_id);
    let XWaitingMessage = getCorrectComponent('XWaitingMessage', v_id);

    // TODO move to CSS
    let top_pane_style = {
      width: '100%',
      position: 'relative',
    };

    let chat_style = {
      width: '100%',
      height: '100%',
      paddingTop: '60px',
      paddingLeft: '20px',
      paddingRight: '20px',
      paddingBottom: '20px',
      overflowY: 'scroll',
    };

    window.setTimeout(() => {
      this.handleResize();
    }, 10);

    top_pane_style['height'] = this.state.chat_height + 'px';

    let wait_message = null;
    if (this.props.chat_state == 'waiting') {
      wait_message = <XWaitingMessage {...this.props} />;
    }

    return (
      <div id="right-top-pane" style={top_pane_style}>
        <ChatNavbar {...this.props} />
        <div id="message-pane-segment" style={chat_style}>
          <XMessageList {...this.props} />
          {wait_message}
        </div>
      </div>
    );
  }
}

class IdleResponse extends React.Component {
  render() {
    return <div id="response-type-idle" className="response-type-module" />;
  }
}

class ReviewButtons extends React.Component {
  GOOD_REASONS = ['Not specified', 'Interesting/Creative', 'Other'];

  BAD_REASONS = [
    'Not specified',
    "Didn't understand task",
    'Bad grammar/spelling',
    'Total nonsense',
    'Slow responder',
    'Other',
  ];

  RATING_VALUES = [1, 2, 3, 4, 5];

  RATING_TITLES = [
    'Terrible',
    'Bad',
    'Average/Good',
    'Great',
    'Above and Beyond',
  ];

  constructor(props) {
    super(props);
    let init_state = props.init_state;
    if (init_state !== undefined) {
      this.state = init_state;
    } else {
      this.state = {
        current_rating: null,
        submitting: false,
        submitted: false,
        text: '',
        dropdown_value: 'Not specified',
      };
    }
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.onInputResize !== undefined) {
      this.props.onInputResize();
    }
  }

  render() {
    // Create basic button selector
    let current_rating = this.state.current_rating;
    let button_vals = this.RATING_VALUES;
    let rating_titles = this.RATING_TITLES;
    let buttons = button_vals.map(v => {
      let use_style = 'info';
      if (v < 3) {
        use_style = 'danger';
      } else if (v > 3) {
        use_style = 'success';
      }

      return (
        <Button
          onClick={() =>
            this.setState({
              current_rating: v,
              text: '',
              dropdown_value: 'Not specified',
            })
          }
          bsStyle={current_rating == v ? use_style : 'default'}
          disabled={this.state.submitting}
          key={'button-rating-' + v}
        >
          {rating_titles[v - 1]}
        </Button>
      );
    });

    // Dropdown and other only appear in some cases
    let dropdown = null;
    let other_input = null;
    let reason_input = null;
    if (current_rating != null && current_rating != 3) {
      let options = current_rating > 3 ? this.GOOD_REASONS : this.BAD_REASONS;
      let dropdown_vals = options.map(opt => (
        <MenuItem
          key={'dropdown-item-' + opt}
          eventKey={opt}
          onSelect={key => this.setState({ dropdown_value: key, text: '' })}
        >
          {opt}
        </MenuItem>
      ));
      dropdown = (
        <DropdownButton
          dropup={true}
          componentClass={InputGroup.Button}
          title={this.state.dropdown_value}
          id={'review-dropdown'}
          disabled={this.state.submitting}
        >
          {dropdown_vals}
        </DropdownButton>
      );
    }

    // Create other text
    if (dropdown != null && this.state.dropdown_value == 'Other') {
      // Optional input for if the user says other
      other_input = (
        <FormControl
          type="text"
          placeholder="Enter reason (optional)"
          value={this.state.text}
          onChange={t => this.setState({ text: t.target.value })}
          disabled={this.state.submitting}
        />
      );
    }
    if (dropdown != null) {
      reason_input = (
        <div style={{ marginBottom: '8px' }}>
          Give a reason for your rating (optional):
          <InputGroup>
            {dropdown}
            {other_input}
          </InputGroup>
        </div>
      );
    }

    // Assemble flow components
    let disable_submit = this.state.submitting || current_rating == null;
    let review_flow = (
      <div>
        Rate your chat partner (fully optional & confidential):
        <br />
        <ButtonGroup>{buttons}</ButtonGroup>
        {reason_input}
        <div style={{ marginBottom: '8px' }}>
          <ButtonGroup style={{ marginBottom: '8px' }}>
            <Button
              disabled={disable_submit}
              bsStyle="info"
              onClick={() => {
                this.setState({ submitting: true });
                let feedback_data = {
                  rating: this.state.current_rating,
                  reason_category: this.state.dropdown_value,
                  reason: this.state.text,
                };
                this.props.onMessageSend(
                  '[PEER_REVIEW]',
                  feedback_data,
                  () => this.setState({ submitted: true }),
                  true // This is a system message, shouldn't be put in feed
                );
                this.props.onChoice(true);
              }}
            >
              {this.state.submitted ? 'Submitted!' : 'Submit Review'}
            </Button>
            <Button
              disabled={this.state.submitting}
              onClick={() => this.props.onChoice(false)}
            >
              Decline Review
            </Button>
          </ButtonGroup>
        </div>
      </div>
    );
    return review_flow;
  }
}

class NextButton extends React.Component {
  // This component is responsible for initiating the click
  // on the next button to get the next subtask from the app

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.onInputResize !== undefined) {
      this.props.onInputResize();
    }
  }
  render() {
    let next_button = (
      <button
        id="next-button"
        type="button"
        className="btn btn-default btn-lg"
        onClick={() => this.props.nextButtonCallback()}
      >
        Next
        <span
          className="glyphicon glyphicon-chevron-right" aria-hidden="true" />{' '}
      </button>
    );

    return (
      <div>
        <div>{next_button}</div>
      </div>
    );
  }
}

class DoneButton extends React.Component {
  // This component is responsible for initiating the click
  // on the mturk form's submit button.

  constructor(props) {
    super(props);
    this.state = {
      feedback_shown: this.props.display_feedback,
      feedback_given: null,
    };
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (this.props.onInputResize !== undefined) {
      this.props.onInputResize();
    }
  }

  render() {
    let review_flow = null;
    let done_button = (
      <button
        id="done-button"
        type="button"
        className="btn btn-primary btn-lg"
        onClick={() => this.props.allDoneCallback()}
      >
        <span className="glyphicon glyphicon-ok-circle" aria-hidden="true" />{' '}
        Done with this HIT
      </button>
    );
    if (this.props.display_feedback) {
      if (this.state.feedback_shown) {
        let XReviewButtons = getCorrectComponent(
          'XReviewButtons',
          this.props.v_id
        );
        review_flow = (
          <XReviewButtons
            {...this.props}
            onChoice={did_give =>
              this.setState({
                feedback_shown: false,
                feedback_given: did_give,
              })
            }
          />
        );
        done_button = null;
      } else if (this.state.feedback_given) {
        review_flow = <span>Thanks for the feedback!</span>;
      }
    }
    return (
      <div>
        {review_flow}
        <div>{done_button}</div>
      </div>
    );
  }
}

class DoneResponse extends React.Component {
  componentDidUpdate(prevProps, prevState, snapshot) {
    this.props.onInputResize();
  }

  render() {
    let v_id = this.props.v_id;
    let XDoneButton = getCorrectComponent('XDoneButton', v_id);
    let XNextButton = getCorrectComponent('XNextButton', v_id);

    let inactive_pane = null;
    if (this.props.done_text) {
      inactive_pane = (
        <span id="inactive" style={{ fontSize: '14pt', marginRight: '15px' }}>
          {this.props.done_text}
        </span>
      );
    }
    // TODO maybe move to CSS?
    let pane_style = {
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
    };
    let button = null;
    if (this.props.task_done) {
      button = <XDoneButton {...this.props} />;
    } else if (this.props.subtask_done && this.props.show_next_task_button) {
      button = <XNextButton {...this.props} />;
    }
    return (
      <div
        id="response-type-done"
        className="response-type-module"
        style={pane_style}
      >
        {inactive_pane}
        {button}
      </div>
    );
  }
}

class TextResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = { textval: '', sending: false };
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $('input#id_text_input').focus();
    }
    this.props.onInputResize();
  }

  tryMessageSend() {
    if (this.state.textval != '' && this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(this.state.textval, {}, () =>
        this.setState({ textval: '', sending: false })
      );
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
      paddingLeft: '25px',
      paddingTop: '20px',
      paddingBottom: '20px',
      paddingRight: '25px',
      float: 'left',
      width: '100%',
    };
    let input_style = {
      height: '50px',
      width: '100%',
      display: 'block',
      float: 'left',
    };
    let submit_style = {
      width: '100px',
      height: '100%',
      fontSize: '16px',
      float: 'left',
      marginLeft: '10px',
      padding: '0px',
    };

    let text_input = (
      <FormControl
        type="text"
        id="id_text_input"
        style={{
          width: '80%',
          height: '100%',
          float: 'left',
          fontSize: '16px',
        }}
        value={this.state.textval}
        placeholder="Please enter here..."
        onKeyPress={e => this.handleKeyPress(e)}
        onChange={e => this.setState({ textval: e.target.value })}
        disabled={!this.props.active || this.state.sending}
      />
    );

    let submit_button = (
      <Button
        className="btn btn-primary"
        style={submit_style}
        id="id_send_msg_button"
        disabled={
          this.state.textval == '' || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Send
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={pane_style}
      >
        <div style={input_style}>
          {text_input}
          {submit_button}
        </div>
      </div>
    );
  }
}

class FormResponse extends React.Component {
  // Provide a form-like interface to MTurk interface.

  constructor(props) {
    super(props);
    // At this point it should be assumed that task_data
    // has a field "respond_with_form"
    let responses = [];
    for (let _ of this.props.task_data['respond_with_form']) {
      responses.push('');
    }
    this.state = { responses: responses, sending: false };
  }

  tryMessageSend() {
    let form_elements = this.props.task_data['respond_with_form'];
    let response_data = [];
    let response_text = '';
    let all_response_filled = true;
    for (let ind in form_elements) {
      let question = form_elements[ind]['question'];
      let response = this.state.responses[ind];
      if (response == '') {
        all_response_filled = false;
      }
      response_data.push({
        question: question,
        response: response,
      });
      response_text += question + ': ' + response + '\n';
    }

    if (all_response_filled && this.props.active && !this.state.sending) {
      this.setState({ sending: true });
      this.props.onMessageSend(
        response_text,
        { form_responses: response_data },
        () => this.setState({ sending: false })
      );
      // clear answers once sent
      this.setState(prevState => {
        prevState['responses'].fill('');
        return { responses: prevState['responses']};
      });
    }
  }

  render() {
    let form_elements = this.props.task_data['respond_with_form'];
    const listFormElements = form_elements.map((form_elem, index) => {
      let question = form_elem['question'];
      if (form_elem['type'] == 'choices') {
        let choices = [<option key="empty_option" />].concat(
          form_elem['choices'].map((option_label, index) => {
            return (
              <option key={'option_' + index.toString()}>{option_label}</option>
            );
          })
        );
        return (
          <FormGroup key={'form_el_' + index}>
            <Col
              componentClass={ControlLabel}
              sm={6}
              style={{ fontSize: '16px' }}
            >
              {question}
            </Col>
            <Col sm={5}>
              <FormControl
                componentClass="select"
                style={{ fontSize: '16px' }}
                value={this.state.responses[index]}
                onChange={e => {
                  var text = e.target.value;
                  this.setState(prevState => {
                    let new_res = prevState['responses'];
                    new_res[index] = text;
                    return { responses: new_res };
                  });
                }}
              >
                {choices}
              </FormControl>
            </Col>
          </FormGroup>
        );
      }
      return (
        <FormGroup key={'form_el_' + index}>
          <Col
            style={{ fontSize: '16px' }}
            componentClass={ControlLabel}
            sm={6}
          >
            {question}
          </Col>
          <Col sm={5}>
            <FormControl
              type="text"
              style={{ fontSize: '16px' }}
              value={this.state.responses[index]}
              onChange={e => {
                var text = e.target.value;
                this.setState(prevState => {
                  let new_res = prevState['responses'];
                  new_res[index] = text;
                  return { responses: new_res };
                });
              }}
            />
          </Col>
        </FormGroup>
      );
    });
    let submit_button = (
      <Button
        className="btn btn-primary"
        style={{ height: '40px', width: '100px', fontSize: '16px' }}
        id="id_send_msg_button"
        disabled={
          this.state.textval == '' || !this.props.active || this.state.sending
        }
        onClick={() => this.tryMessageSend()}
      >
        Send
      </Button>
    );

    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={{
          paddingTop: '15px',
          float: 'left',
          width: '100%',
          backgroundColor: '#eeeeee',
        }}
      >
        <Form
          horizontal
          style={{ backgroundColor: '#eeeeee', paddingBottom: '10px' }}
        >
          {listFormElements}
          <FormGroup>
            <Col sm={6} />
            <Col sm={5}>{submit_button}</Col>
          </FormGroup>
        </Form>
      </div>
    );
  }
}

class ResponsePane extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XDoneResponse = getCorrectComponent('XDoneResponse', v_id);
    let XTextResponse = getCorrectComponent('XTextResponse', v_id);
    let XFormResponse = getCorrectComponent('XFormResponse', v_id);
    let XIdleResponse = getCorrectComponent('XIdleResponse', v_id);

    let response_pane = null;
    switch (this.props.chat_state) {
      case 'done':
      case 'inactive':
        response_pane = <XDoneResponse {...this.props} />;
        break;
      case 'text_input':
      case 'waiting':
        if (this.props.task_data && this.props.task_data['respond_with_form']) {
          response_pane = (
            <XFormResponse
              {...this.props}
              active={this.props.chat_state == 'text_input'}
            />
          );
        } else {
          response_pane = (
            <XTextResponse
              {...this.props}
              active={this.props.chat_state == 'text_input'}
            />
          );
        }
        break;
      case 'idle':
      default:
        response_pane = <XIdleResponse {...this.props} />;
        break;
    }

    return (
      <div
        id="right-bottom-pane"
        style={{ width: '100%', backgroundColor: '#eee' }}
      >
        {response_pane}
      </div>
    );
  }
}

class RightPane extends React.Component {
  handleResize() {
    if (this.chat_pane !== undefined) {
      if (this.chat_pane.handleResize !== undefined) {
        this.chat_pane.handleResize();
      }
    }
  }

  render() {
    let v_id = this.props.v_id;
    let XChatPane = getCorrectComponent('XChatPane', v_id);
    let XResponsePane = getCorrectComponent('XResponsePane', v_id);

    // TODO move to CSS
    let right_pane = {
      minHeight: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'spaceBetween',
    };

    return (
      <div id="right-pane" style={right_pane}>
        <XChatPane
          message_count={this.props.messages.length}
          {...this.props}
          ref={pane => {
            this.chat_pane = pane;
          }}
        />
        <XResponsePane
          {...this.props}
          onInputResize={() => this.handleResize()}
        />
      </div>
    );
  }
}

class ContentPane extends React.Component {
  render () {
    // TODO create some templates maybe? We want to be able to attach
    // pretty robust validation to components, so maybe the idea is
    // to provide a base set of useful components people might want to
    // render or use in their tasks and work from there. We should re-use
    // anything we can from Halo for this.
    return <div>
      If you are seeing this, it is because you haven't defined a custom
      content pane to render your task, and thus it doesn't work yet. See the
      image_captions_demo for an example of how to create this.
    </div>
  }
}

class StaticRightPane extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XContentPane = getCorrectComponent('XContentPane', v_id);

    // TODO move to CSS
    let right_pane = {
      minHeight: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'spaceBetween',
    };

    return (
      <div id="right-pane" style={right_pane}>
        <XContentPane {...this.props} />
      </div>
    );
  }
}

class TaskDescription extends React.Component {
  render() {
    let header_text = CHAT_TITLE;
    let task_desc = this.props.task_description || 'Task Description Loading';
    return (
      <div>
        <h1>{header_text}</h1>
        <hr style={{ borderTop: '1px solid #555' }} />
        <span
          id="task-description"
          style={{ fontSize: '16px' }}
          dangerouslySetInnerHTML={{ __html: task_desc }}
        />
      </div>
    );
  }
}

class ContextView extends React.Component {
  render() {
    // TODO pull context title from templating variable
    let header_text = 'Context';
    let context =
      'To render context here, write or select a ContextView ' +
      'that can render your task_data, or write the desired ' +
      'content into the task_data.html field of your act';
    if (
      this.props.task_data !== undefined &&
      this.props.task_data.html !== undefined
    ) {
      context = this.props.task_data.html;
    }
    return (
      <div>
        <h1>{header_text}</h1>
        <hr style={{ borderTop: '1px solid #555' }} />
        <span
          id="context"
          style={{ fontSize: '16px' }}
          dangerouslySetInnerHTML={{ __html: context }}
        />
      </div>
    );
  }
}

class LeftPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = { current_pane: 'instruction', last_update: 0 };
  }

  static getDerivedStateFromProps(nextProps, prevState) {
    if (
      nextProps.task_data !== undefined &&
      nextProps.task_data.last_update !== undefined &&
      nextProps.task_data.last_update > prevState.last_update
    ) {
      return {
        current_pane: 'context',
        last_update: nextProps.task_data.last_update,
      };
    } else return null;
  }

  render() {
    let v_id = this.props.v_id;
    let frame_height = this.props.frame_height;
    let frame_style = {
      height: frame_height + 'px',
      backgroundColor: '#dff0d8',
      padding: '30px',
      overflow: 'auto',
    };
    let XTaskDescription = getCorrectComponent('XTaskDescription', v_id);
    let pane_size = this.props.is_cover_page ? 'col-xs-12' : 'col-xs-4';
    let has_context = this.props.task_data.has_context;
    if (this.props.is_cover_page || !has_context) {
      return (
        <div id="left-pane" className={pane_size} style={frame_style}>
          <XTaskDescription {...this.props} />
          {this.props.children}
        </div>
      );
    } else {
      let XContextView = getCorrectComponent('XContextView', v_id);
      // In a 2 panel layout, we need to tabulate the left pane to be able
      // to display both context and instructions
      let nav_items = [
        <NavItem
          eventKey={'instruction'}
          key={'instruction-selector'}
          title={'Task Instructions'}
        >
          {'Task Instructions'}
        </NavItem>,
        <NavItem
          eventKey={'context'}
          key={'context-selector'}
          title={'Context'}
        >
          {'Context'}
        </NavItem>,
      ];
      let display_instruction = {
        backgroundColor: '#dff0d8',
        padding: '10px 20px 20px 20px',
        flex: '1 1 auto',
      };
      let display_context = {
        backgroundColor: '#dff0d8',
        padding: '10px 20px 20px 20px',
        flex: '1 1 auto',
      };
      if (this.state.current_pane == 'context') {
        display_instruction.display = 'none';
      } else {
        display_context.display = 'none';
      }
      let nav_panels = [
        <div style={display_instruction} key={'instructions-display'}>
          <XTaskDescription {...this.props} />
        </div>,
        <div style={display_context} key={'context-display'}>
          <XContextView {...this.props} />
        </div>,
      ];

      let frame_style = {
        height: frame_height + 'px',
        backgroundColor: '#eee',
        padding: '10px 0px 0px 0px',
        overflow: 'auto',
        display: 'flex',
        flexFlow: 'column',
      };

      return (
        <div id="left-pane" className={pane_size} style={frame_style}>
          <Nav
            bsStyle="tabs"
            activeKey={this.state.current_pane}
            onSelect={key => this.setState({ current_pane: key })}
          >
            {nav_items}
          </Nav>
          {nav_panels}
          {this.props.children}
        </div>
      );
    }
  }
}

class ContentLayout extends React.Component {
  render() {
    let layout_style = '2-PANEL'; // Currently the only layout style is 2 panel
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XRightPane = getCorrectComponent('XRightPane', v_id);
    return (
      <div className="row" id="ui-content">
        <XLeftPane {...this.props} layout_style={layout_style} />
        <XRightPane {...this.props} layout_style={layout_style} />
      </div>
    );
  }
}

class StaticContentLayout extends React.Component {
  render() {
    let layout_style = '2-PANEL'; // Currently the only layout style is 2 panel
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XStaticRightPane = getCorrectComponent('XStaticRightPane', v_id);
    let XDoneResponse = getCorrectComponent('XDoneResponse', v_id);
    let {frame_height, ...others} = this.props;
    let next_or_done_button = null;
    if (this.props.subtask_done) {
        next_or_done_button = (
          <XDoneResponse {...this.props} onInputResize={() => {}}/>
        )
    }
    return (
      <div className="row" id="ui-content">
        <XLeftPane
          {...others}
          layout_style={layout_style}
          frame_height={frame_height}
        >
          {next_or_done_button}
        </XLeftPane>
        <XStaticRightPane {...this.props} layout_style={layout_style} />
      </div>
    );
  }
}

class BaseFrontend extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XContentLayout = getCorrectComponent('XContentLayout', v_id);

    let content = null;
    if (this.props.is_cover_page) {
      content = (
        <div className="row" id="ui-content">
          <XLeftPane {...this.props} />
        </div>
      );
    } else if (this.props.initialization_status == 'initializing') {
      content = <div id="ui-placeholder">Initializing...</div>;
    } else if (this.props.initialization_status == 'websockets_failure') {
      content = (
        <div id="ui-placeholder">
          Sorry, but we found that your browser does not support WebSockets.
          Please consider updating your browser to a newer version or using
          a different browser and check this HIT again.
        </div>
      );
    } else if (this.props.initialization_status == 'failed') {
      content = (
        <div id="ui-placeholder">
          Unable to initialize. We may be having issues with our servers.
          Please refresh the page, or if that isn't working return the HIT and
          try again later if you would like to work on this task.
        </div>
      );
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

// TODO Require a ContentLayout as a child of BaseFrontend, rather than having
// the component juggle both sets of props and duplicating code between static
// and base frontends
class StaticFrontend extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XLeftPane = getCorrectComponent('XLeftPane', v_id);
    let XStaticContentLayout = getCorrectComponent('XStaticContentLayout', v_id);

    let content = null;
    if (this.props.is_cover_page) {
      content = (
        <div className="row" id="ui-content">
          <XLeftPane {...this.props} />
        </div>
      );
    } else if (this.props.initialization_status == 'initializing') {
      content = <div id="ui-placeholder">Initializing...</div>;
    } else if (this.props.initialization_status == 'websockets_failure') {
      content = (
        <div id="ui-placeholder">
          Sorry, but we found that your browser does not support WebSockets.
          Please consider updating your browser to a newer version or using
          a different browser and check this HIT again.
        </div>
      );
    } else if (this.props.initialization_status == 'failed') {
      content = (
        <div id="ui-placeholder">
          Unable to initialize. We may be having issues with our servers.
          Please refresh the page, or if that isn't working return the HIT and
          try again later if you would like to work on this task.
        </div>
      );
    } else {
      content = <XStaticContentLayout {...this.props} />;
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
  XContentLayout: ['XContentLayout', ContentLayout],
  XLeftPane: ['XLeftPane', LeftPane],
  XRightPane: ['XRightPane', RightPane],
  XResponsePane: ['XResponsePane', ResponsePane],
  XTextResponse: ['XTextResponse', TextResponse],
  XFormResponse: ['XFormResponse', FormResponse],
  XDoneResponse: ['XDoneResponse', DoneResponse],
  XIdleResponse: ['XIdleResponse', IdleResponse],
  XDoneButton: ['XDoneButton', DoneButton],
  XNextButton: ['XNextButton', NextButton],
  XChatPane: ['XChatPane', ChatPane],
  XWaitingMessage: ['XWaitingMessage', WaitingMessage],
  XMessageList: ['XMessageList', MessageList],
  XChatMessage: ['XChatMessage', ChatMessage],
  XTaskDescription: ['XTaskDescription', TaskDescription],
  XReviewButtons: ['XReviewButtons', ReviewButtons],
  XContextView: ['XContextView', ContextView],
  XStaticRightPane: ['XStaticRightPane', StaticRightPane],
  XContentPane: ['XContentPane', ContentPane],
  XStaticContentLayout: ['XStaticContentLayout', StaticContentLayout],
};

export {
  // Original Components
  ChatMessage,
  MessageList,
  ConnectionIndicator,
  Hourglass,
  WaitingMessage,
  ChatPane,
  IdleResponse,
  DoneButton,
  NextButton,
  DoneResponse,
  TextResponse,
  ResponsePane,
  RightPane,
  LeftPane,
  ContentLayout,
  BaseFrontend,
  StaticFrontend,
  // Functions to update and get current components
  setCustomComponents,
  getCorrectComponent,
};
