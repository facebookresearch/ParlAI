/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

/* eslint-disable react/jsx-key */
/* eslint-disable no-unused-vars */

import React from "react";
import { Table, Tooltip } from "antd";
import TrackingSensor from "./TrackingSensor.js";

function _renderExamples(examples, record, renderIndex) {
  return (
    <TrackingSensor name={`example`} event={{ title: record.title }}>
      <ul>
        {(examples || []).map((example, index) => (
          <li key={index}>
            <table style={{ marginBottom: "0.5em" }}>
              {example.content.split("\n").map((line, i) =>
                /(^User:|^System:)/.exec(line) ? (
                  <tr>
                    <td
                      style={{
                        textAlign: "left",
                        verticalAlign: "top",
                        lineHeight: 1.5,
                        color: "rgb(40, 40, 40)",
                        width: "4em",
                        margin: 0
                      }}
                    >
                      {(/(^User:|^System:)/.exec(line) || [null])[0]}&nbsp;
                    </td>
                    <td
                      style={{
                        textAlign: "left",
                        verticalAlign: "top",
                        lineHeight: 1.5,
                        color: "rgb(40, 40, 40)",
                        margin: 0
                      }}
                    >
                      {line.replace(/(^User: |^System: )/, "")}

                      {example.explain !== undefined &&
                      example.explain !== "" &&
                      i + 1 == example.content.split("\n").length ? (
                        <Tooltip
                          title=<TrackingSensor
                            name={`explain-${record.title}-${index}`}
                            event={{ content: example.explain }}
                          >
                            <span>{example.explain}</span>
                          </TrackingSensor>
                        >
                          &nbsp;{" "}
                          <sub>
                            <a>because...</a>
                          </sub>
                        </Tooltip>
                      ) : null}
                    </td>
                  </tr>
                ) : (
                  <>
                    {line}
                    {example.explain !== undefined &&
                    example.explain !== "" &&
                    i + 1 == example.content.split("\n").length ? (
                      <Tooltip title={example.explain}>
                        &nbsp;{" "}
                        <sub>
                          <a>because...</a>
                        </sub>
                      </Tooltip>
                    ) : null}
                  </>
                )
              )}
            </table>
          </li>
        ))}
      </ul>
    </TrackingSensor>
  );
}

function showExamples(questions, styles, titleStyle) {
  if (questions === undefined) {
    return null;
  }
  let columnsExample = [
    {
      title: "Question",
      dataIndex: "title",
      width: "15%"
    },
    {
      title: "Instructions",
      dataIndex: "showInstructionFn",
      key: "showInstructionFn",
      render: showInstructionFn => showInstructionFn(),
      width: "40%"
    }
  ];

  let hasExp = false;
  for (const question of questions) {
    if (question.type == "Radio") {
      for (let i = 0; i < question.options.length; i += 1) {
        if ((question[`exampleOpt${i}`] || []).length > 0) {
          hasExp = true;
        }
      }
    } else {
      if (
        (question.examples || []).length > 0 ||
        (question.counterexamples || []).length > 0
      ) {
        hasExp = true;
      }
    }
  }

  if (hasExp) {
    columnsExample.push({
      title: "Example",
      dataIndex: "showExampleFn",
      key: "showExampleFn",
      render: showExampleFn => showExampleFn(),
      width: "40%"
    });
  }

  let dsExamples = [];
  for (const question of questions) {
    let showInstructionFn;
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
        if (instructionOpts.length == 0) {
          showInstructionFn = () => question.instruction;
        } else {
          showInstructionFn = () => (
            <>
              <p>{question.instruction}</p>
              <span>
                <ul>{instructionOpts}</ul>
              </span>
            </>
          );
        }
      }
    } else {
      showInstructionFn = () => question.instruction;
    }

    let showExampleFn;
    if (question.type === "Likert Scale") {
      if (
        (question.examples || []).length === 0 &&
        (question.counterexamples || []).length === 0
      ) {
        continue;
      } else {
        showExampleFn = () => (
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
    } else if (question.type === "Radio") {
      showExampleFn = () =>
        question.options.map((opt, i) => (
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
        ));
    } else {
      if (
        (question.examples || []).length === 0 &&
        (question.counterexamples || []).length === 0
      ) {
        continue;
      } else {
        showExampleFn = () => (
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

    dsExamples.push({
      title: question.title,
      instruction: question.instruction,
      showExampleFn,
      showInstructionFn
    });
  }
  if (dsExamples.length === 0) {
    return null;
  } else {
    return (
      <>
        <div style={titleStyle}>
          <b>Examples</b>
        </div>
        <Table
          rowKey="sentid"
          dataSource={dsExamples}
          columns={columnsExample}
          size="small"
          pagination={{ hideOnSinglePage: true }}
          style={styles.example}
        />
      </>
    );
  }
}

/* eslint-enable react/jsx-key */
/* eslint-enable no-unused-vars */

export { _renderExamples, showExamples };
