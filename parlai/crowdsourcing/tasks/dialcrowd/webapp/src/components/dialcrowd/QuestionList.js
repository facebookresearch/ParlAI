/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React from "react";
import { Form, Input } from "antd";
import ReactMarkdown from "react-markdown";

function showFeedbackQuestion(show) {
  /* Show feedback question to the workers
   */
  if (!show) {
    return null;
  }
  return (
    <div
      title={"Feedback"}
      style={{ width: "500px", margin: "0 auto", "text-align": "center" }}
    >
      <p style={{ fontSize: 15, color: "black" }}>
        <b>(Optional)</b> Please let us know if you have feedback:
      </p>
      <Form.Item name="FeedbackOpen|||1">
        <Input placeholder="" />
      </Form.Item>
    </div>
  );
}

class Markdown extends React.Component {
  /* Props:
     {@Bool} enableMarkdown
  */
  /* eslint-disable react/no-children-prop */
  render() {
    if (this.props.enableMarkdown) {
      return <ReactMarkdown children={this.props.children} />;
    } else {
      return <> {this.props.children} </>;
    }
  }
  /* eslint-enable react/no-children-prop */
}

export { showFeedbackQuestion, Markdown };
