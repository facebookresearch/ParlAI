/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React, { Component } from "react";
import { Button, Form, Radio, Select, Modal } from "antd";
import QuestionList from "../../workers/QuestionList.js";
import { ConsentForm } from "../../workers/AgreeModal.js";
import {
  _renderExamples,
  showExamples
} from "../../workers/worker_category.js";

/* eslint-disable react/no-unescaped-entities  */
/* eslint-disable no-unused-vars  */

class PreviewConsent extends Component {
  /* Props
   * @{String} consent: PDF source of the consent form document.
   * @{Array} requirements: [{key: @{int}, content: @{string}}]
   */
  constructor(props) {
    super(props);
    this.state = { visible: false };
  }

  render() {
    const formItemLayoutWithOutLabel = {
      wrapperCol: {
        xs: { span: 24, offset: 0 },
        sm: { span: 20, offset: 4 }
      }
    };

    return (
      <>
        <Form.Item {...formItemLayoutWithOutLabel}>
          <Button
            onClick={() => {
              this.showPreview();
            }}
            style={{ width: "90%" }}
            disabled={
              this.props.requirements.length === 0 && this.props.consent === ""
            }
          >
            Preview the Above Consent Questions in Worker View
          </Button>
        </Form.Item>
        {this.state.visible ? (
          <ConsentForm
            consent={this.props.consent}
            checkboxes={this.props.requirements}
            forceShow={this.state.visible}
            onAccept={() => this.close()}
            config={true}
          />
        ) : null}
      </>
    );
  }

  showPreview() {
    this.setState({ visible: true });
  }

  close() {
    this.setState({ visible: false });
  }
}

class PreviewExample extends Component {
  /* Params
     @{Object}	questions
   */
  constructor(props) {
    super(props);
    this.state = { visible: false };
  }

  render() {
    const columns = [
      {
        title: "Question",
        dataIndex: "title"
      },
      {
        title: "Examples",
        dataIndex: "examples",
        key: "examples",
        render: _renderExamples
      },
      {
        title: "Counterexamples",
        dataIndex: "counterexamples",
        key: "counterexamples",
        render: _renderExamples
      }
    ];

    return (
      <>
        <Button
          onClick={() => {
            this.showPreview();
          }}
          style={{ width: "90%" }}
        >
          Preview the Examples and Counterexamples in Worker View
        </Button>
        <Modal
          visible={this.state.visible}
          centered
          zIndex={1000}
          onCancel={() => {
            this.close();
          }}
          footer={
            <Button
              onClick={() => {
                this.close();
              }}
            >
              {" "}
              Close{" "}
            </Button>
          }
          width={"75%"}
        >
          {showExamples(this.props.questions, {}, {})}
        </Modal>
      </>
    );
  }

  showPreview() {
    this.setState({ visible: true });
  }

  close() {
    this.setState({ visible: false });
  }
}

class PreviewQuestionInner extends Component {
  /* Params
     @{Object}	question
     @{Array}	systemNames: For A/B test.
   */
  constructor(props) {
    super(props);
    this.state = { visible: false };
  }

  render() {
    return (
      <>
        <Button
          onClick={() => {
            this.showPreview();
          }}
          style={{ width: "90%" }}
        >
          Preview this Question in Worker View
        </Button>
        <Modal
          visible={this.state.visible}
          centered
          zIndex={1000}
          onCancel={() => {
            this.close();
          }}
          footer={
            <Button
              onClick={() => {
                this.close();
              }}
            >
              {" "}
              Close{" "}
            </Button>
          }
          width={"60%"}
        >
          <QuestionList
            questions={[this.props.question]}
            title=""
            fieldPrefix=""
            systemNames={this.props.systemNames}
            borderStyle="none"
          />
        </Modal>
      </>
    );
  }

  showPreview() {
    this.setState({ visible: true });
  }

  close() {
    this.setState({ visible: false });
  }
}

class PreviewIntent extends Component {
  /* Params
     @{Object}	questions
   */
  constructor(props) {
    super(props);
    this.state = { visible: false };
  }

  render() {
    const formItemLayout2 = {
      labelCol: { span: 14 },
      wrapperCol: { span: 10 },
      colon: false
    };

    return (
      <>
        <Button
          onClick={() => {
            this.showPreview();
          }}
          style={{ width: "90%" }}
        >
          Preview in Worker View
        </Button>
        <Modal
          visible={this.state.visible}
          centered
          zIndex={1000}
          onCancel={() => {
            this.close();
          }}
          footer={
            <Button
              onClick={() => {
                this.close();
              }}
            >
              {" "}
              Close{" "}
            </Button>
          }
          width={"75%"}
        >
          <div title="category classification">
            <div style={{ textAlign: "center" }}>
              <p style={{ textAlign: "center", fontSize: 18 }}>
                <b>"Select a category for the given text"</b>
              </p>
            </div>
            <div style={{ textAlign: "center" }}>
              <Button type="default" onClick={this.openInstructions}>
                Example Responses
              </Button>
            </div>
            <div style={{ backgroundColor: "#f7f7f7" }}>
              <Form.Item
                className={"two-rows-label"}
                {...formItemLayout2}
                label={
                  <div
                    style={{
                      display: "inline-block",
                      float: "left",
                      whiteSpace: "normal",
                      marginRight: "12px",
                      "text-align": "left",
                      lineHeight: "15px"
                    }}
                  >
                    <p>Some text here.</p>
                  </div>
                }
                hasFeedback
              >
                {/* eslint-disable react/jsx-key */}
                <Select placeholder="Please select a category">
                  {this.props.questions.map((question, j) => (
                    <Select.Option value={question.title}>
                      {question.title}
                    </Select.Option>
                  ))}
                </Select>
                {/* eslint-enable react/jsx-key */}
              </Form.Item>
              <Form.Item
                {...formItemLayout2}
                label={
                  <div
                    style={{
                      color: "forestgreen",
                      display: "inline-flex"
                    }}
                  >
                    {
                      "Confidence of your answer? (1: Not confident, 3: Very confident) "
                    }
                  </div>
                }
              >
                <Radio.Group>
                  <Radio value="1">1</Radio>
                  <Radio value="2">2</Radio>
                  <Radio value="3">3</Radio>
                </Radio.Group>
              </Form.Item>
            </div>
          </div>
        </Modal>
      </>
    );
  }

  showPreview() {
    this.setState({ visible: true });
  }

  close() {
    this.setState({ visible: false });
  }
}

// const PreviewQuestion = Form.create()(PreviewQuestionInner);
const PreviewQuestion = PreviewQuestionInner;

/* eslint-enable react/no-unescaped-entities  */
/* eslint-enable no-unused-vars */

export default PreviewQuestion;
export { PreviewIntent, PreviewQuestion, PreviewConsent, PreviewExample };
