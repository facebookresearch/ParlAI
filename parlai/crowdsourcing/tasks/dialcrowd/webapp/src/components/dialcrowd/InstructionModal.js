/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React from "react";
import { Button, Modal, Divider, Tooltip } from "antd";
import { QuestionCircleOutlined } from "@ant-design/icons";

/* eslint-disable react/jsx-key */

class InstructionModal extends React.Component {
  /* Render instruction panel for worker
   * Props
   * @{bool} initState: Whether the window is visible or not.
   * question
   */
  constructor(props) {
    super(props);
    this.state = {
      visible: this.props.initState || false
    };
  }

  render() {
    const visible = this.state.visible || this.state.forceShow;

    const footer = (
      <Button key="accept" type="primary" closable="false" onClick={this.close}>
        Close
      </Button>
    );

    return (
      <>
        <a
          onClick={() => {
            this.setState({ visible: true });
          }}
          style={{ fontSize: "0.8em" }}
        >
          <QuestionCircleOutlined />
          &nbsp; show instruction
        </a>
        <style>
          {`
        .ant-modal-content {height: 100%; display: flex; flex-direction: column;}
          `}
        </style>

        <Modal
          visible={visible}
          title={this.props.question.title}
          width="80%"
          closable={true}
          maskClosable={false}
          bodyStyle={{ flexGrow: 1 }}
          centered
          zIndex={1000}
          footer={footer}
          onCancel={this.close}
        >
          {this.renderInstruction()}
        </Modal>
      </>
    );
  }

  renderInstruction() {
    const styleInst = {
      color: "black",
      fontSize: 18
    };
    const styleExp = {
      color: "black",
      fontSize: 16
    };

    return (
      <>
        <div style={styleInst}>{showInstruction(this.props.question)}</div>
        <div style={styleExp}>{showExamples(this.props.question)}</div>
      </>
    );
  }
  close = () => {
    this.setState({ visible: false });
  };
}

function showInstruction(question) {
  if (question.type == "Radio") {
    let instructionOpts = [];
    for (let i = 0; i < (question.options || []).length; i += 1) {
      if (question[`instructionOpt${i}`] !== undefined) {
        instructionOpts.push(
          <li>
            <b>
              <i>{question.options[i].content}</i>
            </b>
            : &nbsp;
            {question[`instructionOpt${i}`]}
          </li>
        );
      }
    }

    if (instructionOpts.length == 0) {
      return question.instruction;
    } else {
      return (
        <>
          <p>{question.instruction}</p>
          <span>
            <ul>{instructionOpts}</ul>
          </span>
        </>
      );
    }
  } else {
    return question.instruction;
  }
}

function showExamples(question) {
  switch (question.type) {
    case "Likert Scale": {
      if (
        (question.examples || []).length === 0 &&
        (question.counterexamples || []).length === 0
      ) {
        return null;
      } else {
        return (
          <>
            <div>
              Examples that strongly agree with the description:
              <ul>
                {(question.examples || []).map(exp => (
                  <li>
                    {exp.content}
                    {exp.explain !== undefined ? (
                      <Tooltip title={exp.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              Examples that strongly disagree with the description:
              <ul>
                {(question.counterexamples || []).map(exp => (
                  <li>
                    {exp.content}
                    {exp.explain !== undefined ? (
                      <Tooltip title={exp.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          </>
        );
      }
    }

    case "Radio": {
      let noExp = true;
      for (let i = 0; i < question.options.length; i += 1) {
        if ((question[`exampleOpt${i}`] || []).length > 0) {
          noExp = false;
        }
      }
      if (noExp) {
        return null;
      }
      return (
        <>
          <Divider> Examples </Divider>
          {question.options.map((opt, i) => (
            <div>
              Examples for choice{" "}
              <i>
                <b> {opt.content} </b>
              </i>
              <ul>
                {(question[`exampleOpt${i}`] || []).map(exp => (
                  <li>
                    {exp.content}
                    {exp.explain !== undefined ? (
                      <Tooltip title={exp.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </>
      );
    }

    default: {
      if (
        (question.examples || []).length === 0 &&
        (question.counterexamples || []).length === 0
      ) {
        return null;
      } else {
        return (
          <>
            <div>
              Examples:
              <ul>
                {question.examples.map(exp => (
                  <li>
                    {exp.content}
                    {exp.explain !== undefined ? (
                      <Tooltip title={exp.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              Counterexamples:
              <ul>
                {question.counterexamples.map(exp => (
                  <li>
                    {exp.content}
                    {exp.explain !== undefined ? (
                      <Tooltip title={exp.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          </>
        );
      }
    }
  }
}

/* eslint-enable react/jsx-key */

export { InstructionModal };
