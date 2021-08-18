/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { Col, Row } from "react-bootstrap";

import { Checkboxes } from './checkboxes.jsx';


function MaybeCheckboxChatMessage({ isSelf, duration, agentName, message = "", checkbox = null }) {
  const floatToSide = isSelf ? "right" : "left";
  const alertStyle = isSelf ? "alert-info" : "alert-warning";

  return (
    <div className="row" style={{ marginLeft: "0", marginRight: "0" }}>
      <div
        className={"alert message " + alertStyle}
        role="alert"
        style={{ float: floatToSide }}
      >
        <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }}>
          <b>{agentName}</b>: <span dangerouslySetInnerHTML={{ __html: message }}></span>
        </span>
        {checkbox}
      </div>
    </div>
  );
}


function PersonaLines({taskData, isLeftPane}) {
  var remarks = null;
  if (isLeftPane === false) {
    remarks = (
      <div>
       <ul>
         <li>Play your given role with respect to those facts mentioned (do not act as yourself).</li>
         <li><b>AVOID repeating facts</b> that the other speaker already mentioned last time.</li>
       </ul>
       </div>
    )
  }
  if (taskData !== undefined && taskData.personas !==undefined) {
    return (
     <div> 
      <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }}>
        Assume <span style={{color: "#0869af", fontWeight: "bold"}}>YOU </span> and the other speaker (<span style={{color: "#a7a050", fontWeight: "bold" }}>THEY </span>) spoke {taskData.time_num} {taskData.time_unit} ago. Please <b>expand on the OPENING TOPIC as MANY TURNs as you can with MORE DETAILS</b>. 
      </span>
      <h5>Below is a most up-to-date summary of the facts you two have mentioned to each other before.</h5>
       {remarks}
        <Row>
          <Col sm={5} style={{ padding: "2px", margin: "5px" }}>
          <div
            className={"alert " + "alert-warning" }
            role="alert"
            style={{ float: "left", display: "table", width: "100%", margin: "10px", padding: 10 }}
          >
            <b><i>THEY</i> mentioned last time</b> : {taskData.personas[0]}
          </div>
        </Col>
        <Col sm={1} style={{ padding: "0px", margin: "0px" }} />
        <Col sm={5} style={{ padding: "2px", margin: "5px" }}>
          <div
            className={"alert " +"alert-info"}
            role="alert"
            style={{ float: "right", display: "table", width: "100%", margin: "10px", padding: 10 }}
          >
            <b><i>YOU</i> mentioned last time</b>: {taskData.personas[1]}
          </div>
        </Col>
        </Row>
      </div>
      )
  } else {
    return null;
  }
}

function CoordinatorChatMessage({ agentName, message = "", taskData}) {
  const floatToSide = "left";
  const alertStyle  = "alert-success";

  return (
    <div className="row" style={{ marginLeft: "0", marginRight: "0" }}>
      <div
        className={"alert message " + alertStyle}
        role="alert"
        style={{ float: floatToSide }}
      >
        <span style={{ fontSize: "16px", whiteSpace: "pre-wrap" }}>
          <b>{agentName}</b>: {message}
        </span>
        <PersonaLines
        taskData={taskData}
        isLeftPane={false}
        />
      </div>
    </div>
  );
}

function RenderChatMessage({ message, mephistoContext, appContext, idx }) {
  const { agentId, taskConfig } = mephistoContext;
  const { currentAgentNames } = appContext.taskContext;
  const { appSettings, setAppSettings } = appContext;
  const { checkboxValues } = appSettings;
  const isHuman = (message.id === agentId || message.id == currentAgentNames[agentId]);
  const annotationBuckets = taskConfig.annotation_buckets;
  const annotationIntro = taskConfig.annotation_question;

  if (message.id === 'Coordinator'){
    return (
      <div>
        <CoordinatorChatMessage
          agentName={
            message.id in currentAgentNames
              ? currentAgentNames[message.id]
              : message.id
          }
          message={message.text}
          taskData={message.task_data}
          messageId={message.message_id}
        />
      </div>
    ); 
  }
  
  if (message.id === 'SUBMIT_WORLD_DATA' || message.id == 'System') {
    return <div />;
  }
  
  var checkboxes = null;
  if (!isHuman && annotationBuckets !== null) {
    let thisBoxAnnotations = checkboxValues[idx];
    if (!thisBoxAnnotations) {
      thisBoxAnnotations = Object.fromEntries(
        annotationBuckets.map(bucket => [bucket.value, false])
      )
    }
    checkboxes = <div style={{"fontStyle": "italic"}}>
      <br />
      {/* {annotationIntro} */}
      <span style={{color: 'green'}}>What piece of previous chat history does this comment from your parnter (THEY) <b> correctly recall or pay attention to</b>? </span>
      <span style={{color: 'blue'}}> And is it <b>engaging</b>?</span> (Check all that apply)
      <br />
      <Checkboxes 
        annotations={thisBoxAnnotations} 
        onUpdateAnnotations={
          (newAnnotations) => {
            checkboxValues[idx] = newAnnotations;
            setAppSettings({checkboxValues});
          }
        } 
        annotationBuckets={annotationBuckets} 
        turnIdx={idx} 
        askReason={false} 
        enabled={idx == appSettings.numMessages - 1}
      />
    </div>;
  }
  return (
    <MaybeCheckboxChatMessage
      isSelf={isHuman}
      agentName={
        message.id in currentAgentNames
          ? currentAgentNames[message.id]
          : message.id
      }
      message={message.text}
      taskData={message.task_data}
      messageId={message.message_id}
      checkbox={checkboxes}
    />
  );
}

export { RenderChatMessage, MaybeCheckboxChatMessage, PersonaLines };