/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import {
  Button,
  Col,
  ControlLabel,
  Form,
  FormControl,
  FormGroup,
  Grid,
  Input,
  Label,
  Radio,
  Row
} from "react-bootstrap";
import { getCorrectComponent } from "./core_components.jsx";
import $ from "jquery";
import Slider, { Range } from "rc-slider";

// blue
const speaker1_color = "#29BFFF";
// purple
const speaker2_color = "#492FED";
// grey
const otherspeaker_color = "#eee";

const speaker1_style = {
  borderRadius: 3,
  padding: "1px 4px",
  display: "inline-block",
  backgroundColor: speaker1_color,
  color: "white"
};
const speaker2_style = {
  borderRadius: 3,
  padding: "1px 4px",
  display: "inline-block",
  backgroundColor: speaker2_color,
  color: "white"
};
const otherspeaker_style = {
  borderRadius: 3,
  padding: "1px 4px",
  display: "inline-block",
  backgroundColor: otherspeaker_color
};

class ChatMessage extends React.Component {
  render() {
    let message = this.props.message;
    let primary_speaker_color = this.props.conv_ind
      ? speaker2_color
      : speaker1_color;
    let message_container_style = {
      display: "block",
      width: "100%",
      ...(this.props.is_primary_speaker
        ? {
            float: "left"
          }
        : {
            float: "right"
          })
    };
    let message_style = {
      borderRadius: 6,
      marginBottom: 10,
      padding: "5px 10px",
      ...(this.props.is_primary_speaker
        ? {
            marginRight: 20,
            textAlign: "left",
            float: "left",
            color: "white",
            display: "inline-block",
            backgroundColor: primary_speaker_color
          }
        : {
            textAlign: "right",
            float: "right",
            display: "inline-block",
            marginLeft: 20,
            backgroundColor: otherspeaker_color
          })
    };
    return (
      <div style={message_container_style}>
        <div style={message_style}>{message}</div>
      </div>
    );
  }
}

class MessageList extends React.Component {
  makeMessages() {
    let agent_id = this.props.agent_id;
    if (this.props.task_data.pairing_dict === undefined) {
      return (
        <div>
          <p> Loading chats </p>
        </div>
      );
    }
    let task_data = this.props.task_data;
    let conv_order_ind =
      task_data.task_specs.conversation_order[this.props.index];
    let messages =
      task_data.pairing_dict.dialogue_dicts[conv_order_ind].dialogue;
    console.log("MESSAGES");
    console.log(task_data.pairing_dict.dialogue_dicts[conv_order_ind]);
    console.log(messages);
    let primary_speaker =
      task_data.pairing_dict.speakers_to_eval[conv_order_ind];

    let XChatMessage = getCorrectComponent("XChatMessage", this.props.v_id);
    let onClickMessage = this.props.onClickMessage;
    if (typeof onClickMessage !== "function") {
      onClickMessage = idx => {};
    }
    return messages.map((m, idx) => (
      <div key={conv_order_ind + "_" + idx} onClick={() => onClickMessage(idx)}>
        <XChatMessage
          message={m.text}
          is_primary_speaker={m.speaker == primary_speaker}
          conv_ind={this.props.index}
        />
      </div>
    ));
  }

  render() {
    return (
      <div id="message_thread" style={{ width: "100%" }}>
        {this.makeMessages()}
      </div>
    );
  }
}

class ChatPane extends React.Component {
  constructor(props) {
    super(props);
    this.state = { chat_height: this.getChatHeight() };
  }

  getChatHeight() {
    let entry_pane = $("div#right-bottom-pane").get(0);
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
    let XMessageList = getCorrectComponent("XMessageList", v_id);

    // TODO move to CSS
    let top_pane_style = {
      width: "100%",
      position: "relative",
      overflowY: "scroll"
    };

    let chat_style = {
      width: "100%",
      height: "100%",
      paddingTop: "60px",
      paddingLeft: "20px",
      paddingRight: "20px",
      paddingBottom: "20px",
      overflowY: "scroll"
    };

    window.setTimeout(() => {
      this.handleResize();
    }, 10);

    top_pane_style["height"] = this.state.chat_height + "px";

    return (
      <div id="right-top-pane" style={top_pane_style}>
        <Grid className="show-grid" style={{ width: "auto" }}>
          <Row>
            <Col sm={6}>
              <div id="message-pane-segment-left" style={chat_style}>
                <XMessageList {...this.props} index={0} />
              </div>
            </Col>
            <Col sm={6}>
              <div id="message-pane-segment-right" style={chat_style}>
                <XMessageList {...this.props} index={1} />
              </div>
            </Col>
          </Row>
        </Grid>
      </div>
    );
  }
}

class EvalResponse extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      speakerChoice: "",
      textReason: "",
      taskData: [],
      subtaskIndexSeen: 0
    };
    this.handleInputChange = this.handleInputChange.bind(this);
    this.handleEnterKey = this.handleEnterKey.bind(this);
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    // Only change in the active status of this component should cause a
    // focus event. Not having this would make the focus occur on every
    // state update (including things like volume changes)
    if (this.props.active && !prevProps.active) {
      $("input#id_text_input").focus();
    }
    this.props.onInputResize();
  }

  checkValidData() {
    console.log(this.state);
    if (this.state.speakerChoice !== "") {
      let response_data = {
        speakerChoice: this.state.speakerChoice,
        textReason: this.state.textReason
      };
      this.props.onValidDataChange(true, response_data);
      return;
    }
    this.props.onValidDataChange(false, {});
  }

  handleInputChange(event) {
    console.log(event);
    let target = event.target;
    let value = target.value;
    let name = target.name;

    this.setState({ [name]: value }, this.checkValidData);
  }

  handleEnterKey(event) {
    event.preventDefault();
    if (this.props.task_done) {
      this.props.allDoneCallback();
    } else if (this.props.subtask_done && this.props.show_next_task_button) {
      this.props.nextButtonCallback();
    }
  }

  render() {
    if (
      this.props.current_subtask_index != null &&
      this.props.current_subtask_index !== this.state.subtaskIndexSeen
    ) {
      this.setState({
        subtaskIndexSeen: this.props.current_subtask_index,
        textReason: "",
        speakerChoice: ""
      });
    }
    if (
      this.props.task_data === undefined ||
      this.props.task_data.task_specs === undefined
    ) {
      return <div></div>;
    }
    let s1_choice = this.props.task_data.task_specs.s1_choice.split(
      "<Speaker 1>"
    );
    let s2_choice = this.props.task_data.task_specs.s2_choice.split(
      "<Speaker 2>"
    );
    let s1_name, s2_name;
    if (this.props.task_data.task_specs.conversation_order[0] == 0) {
      s1_name = this.props.task_data.pairing_dict.speakers_to_eval[0];
      s2_name = this.props.task_data.pairing_dict.speakers_to_eval[1];
    } else {
      s1_name = this.props.task_data.pairing_dict.speakers_to_eval[1];
      s2_name = this.props.task_data.pairing_dict.speakers_to_eval[0];
    }
    let form_question = this.props.task_data.task_specs.question;
    let text_question =
      "Please provide a brief justification for your choice (a few words or a sentence)";
    let text_reason = (
      <div>
        <ControlLabel>{text_question}</ControlLabel>
        <FormControl
          type="text"
          id="id_text_input"
          name="textReason"
          style={{
            width: "80%",
            height: "100%",
            float: "left",
            fontSize: "16px"
          }}
          value={this.state.textReason}
          placeholder="Please enter here..."
          onChange={this.handleInputChange}
        />
      </div>
    );
    let speaker1_div = <div style={speaker1_style}>Speaker 1</div>;
    let speaker2_div = <div style={speaker2_style}>Speaker 2</div>;
    let choice1 = (
      <div>
        {s1_choice[0]}
        {speaker1_div}
        {s1_choice[1]}
      </div>
    );
    let choice2 = (
      <div>
        {s2_choice[0]}
        {speaker2_div}
        {s2_choice[1]}
      </div>
    );
    return (
      <div
        id="response-type-text-input"
        className="response-type-module"
        style={{
          paddingTop: "15px",
          float: "left",
          width: "100%",
          backgroundColor: "#eeeeee"
        }}
      >
        <Form
          horizontal
          style={{ backgroundColor: "#eeeeee", paddingBottom: "10px" }}
          onSubmit={this.handleEnterKey}
        >
          <div className="container" style={{ width: "auto" }}>
            <ControlLabel> {form_question} </ControlLabel>
            <FormGroup>
              <Col sm={6}>
                <Radio
                  name="speakerChoice"
                  value={s1_name}
                  style={{ width: "100%" }}
                  checked={this.state.speakerChoice == s1_name}
                  onChange={this.handleInputChange}
                >
                  {choice1}
                </Radio>
              </Col>
              <Col sm={6}>
                <Radio
                  name="speakerChoice"
                  value={s2_name}
                  style={{ width: "100%" }}
                  checked={this.state.speakerChoice == s2_name}
                  onChange={this.handleInputChange}
                >
                  {choice2}
                </Radio>
              </Col>
            </FormGroup>
            {text_reason}
          </div>
        </Form>
      </div>
    );
  }
}

class ResponsePane extends React.Component {
  render() {
    let v_id = this.props.v_id;
    let XDoneResponse = getCorrectComponent("XDoneResponse", v_id);
    let XEvalResponse = getCorrectComponent("XEvalResponse", v_id);

    let response_pane = null;
    switch (this.props.task_state) {
      case "done":
        response_pane = <XDoneResponse {...this.props} />;
        break;
      default:
        response_pane = <XEvalResponse {...this.props} />;
        break;
    }

    return (
      <div
        id="right-bottom-pane"
        style={{ width: "100%", backgroundColor: "#eee" }}
      >
        {response_pane}
      </div>
    );
  }
}

class PairwiseEvalPane extends React.Component {
  handleResize() {
    console.log("HANDLE RESIZE CALLED");
    if (this.chat_pane !== undefined && this.chat_pane !== null) {
      console.log(this.chat_pane);
      if (this.chat_pane.handleResize !== undefined) {
        console.log(this.chat_pane.handleResize);
        this.chat_pane.handleResize();
      }
    }
  }

  render() {
    let v_id = this.props.v_id;
    let XChatPane = getCorrectComponent("XChatPane", v_id);
    let XResponsePane = getCorrectComponent("XResponsePane", v_id);
    let XTaskFeedbackPane = getCorrectComponent("XTaskFeedbackPane", v_id);

    let right_pane = {
      maxHeight: "60%",
      display: "flex",
      flexDirection: "column",
      justifyContent: "spaceBetween",
      width: "auto"
    };
    if (
      this.props.current_subtask_index >=
      this.props.task_description.num_subtasks
    ) {
      return (
        <div id="right-pane" style={right_pane}>
          <XTaskFeedbackPane
            {...this.props}
            ref={pane => {
              this.chat_pane = pane;
            }}
            onInputResize={() => this.handleResize()}
          />
        </div>
      );
    }
    console.log("RETURNING");
    return (
      <div id="right-pane" style={right_pane}>
        <XChatPane
          {...this.props}
          message_count={this.props.messages.length}
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

class TaskDescription extends React.Component {
  render() {
    let header_text = "Which Conversational Partner is Better?";
    if (this.props.task_description === null) {
      return <div>Loading</div>;
    }
    let num_subtasks = this.props.task_description.num_subtasks;

    let question = this.props.task_description.question;
    let content = (
      <div>
        In this task, you will read two conversations and judge&nbsp;
        <div style={speaker1_style}>Speaker 1</div> on the left and&nbsp;
        <div style={speaker2_style}>Speaker 2</div> on the right&nbsp; based on
        the quality of conversation only.{" "}
        <b>Don't base your judgement&nbsp; on their hobbies, job, etc.</b>&nbsp;
        Do your best to ignore the{" "}
        <div style={otherspeaker_style}>other speaker</div>.&nbsp; You may need
        to scroll down to see the full conversations.&nbsp;
        <br />
        <br />
        You will judge <div style={speaker1_style}>Speaker 1</div> and&nbsp;
        <div style={speaker2_style}>Speaker 2</div> on this:&nbsp;
        <b>{question}</b> You should&nbsp; also provide a very brief
        justification. Failure to do so could result&nbsp; in your hits being
        rejected.
        <br />
        <br />
        <b>
          {" "}
          You will do this for {num_subtasks} pairs of conversations.&nbsp; Use
          the [NEXT] button when you're done with each judgment.
        </b>
        <br />
        <br />
        NOTE: please be sure to only accept one of this task at a time.&nbsp;
        Additional pages will show errors or fail to load and you wll not be
        able to submit the hit.&nbsp;
        <h4>Please accept the task if you're ready.</h4>
      </div>
    );
    if (!this.props.is_cover_page) {
      if (this.props.task_data.task_specs === undefined) {
        return <div>Loading</div>;
      }
      let num_subtasks = this.props.num_subtasks;
      let cur_index = this.props.current_subtask_index + 1;
      let question = this.props.task_data.task_specs.question;
      content = (
        <div>
          <b>
            You are currently at comparison {cur_index} / {num_subtasks}{" "}
          </b>
          <br />
          <br />
          You will read two conversations and judge&nbsp;
          <div style={speaker1_style}>Speaker 1</div> on the left and&nbsp;
          <div style={speaker2_style}>Speaker 2</div> on the right&nbsp; based
          on the quality of conversation.{" "}
          <b>Don't base your judgement&nbsp; on their hobbies, job, etc. </b>
          &nbsp; Do your best to ignore the{" "}
          <div style={otherspeaker_style}>other speaker</div>.&nbsp; You may
          need to scroll down to see the full conversations.&nbsp;
          <br />
          <br />
          You will judge <div style={speaker1_style}>Speaker 1</div> and&nbsp;
          <div style={speaker2_style}>Speaker 2</div> on this:&nbsp;
          <b>{question}</b> You should&nbsp; also provide a very brief
          justification. Failure to do so could result&nbsp; in your hits being
          rejected.
          <br />
          <br />
          <b>
            {" "}
            You will do this for {num_subtasks} pairs of conversations.&nbsp;
            After completing each judgement, use the [NEXT] button (which will
            appear below after you finish your judgement).
          </b>
          <br />
        </div>
      );
    }
    return (
      <div>
        <h1>{header_text}</h1>
        <hr style={{ borderTop: "1px solid #555" }} />
        {content}
      </div>
    );
  }
}

export default {
  XChatMessage: { default: ChatMessage },
  XMessageList: { default: MessageList },
  XChatPane: { default: ChatPane },
  XEvalResponse: { default: EvalResponse },
  XResponsePane: { default: ResponsePane },
  XContentPane: { default: PairwiseEvalPane },
  XTaskDescription: { default: TaskDescription }
};
