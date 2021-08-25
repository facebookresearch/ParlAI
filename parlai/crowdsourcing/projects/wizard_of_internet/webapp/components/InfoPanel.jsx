/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";

export default function InfoPanel({ isWizard, personaDesc }) {
  return (<div id="info-bar">
    <TaskDescription
      isWizard={isWizard} />
    <Persona
      isWizard={isWizard}
      personaDesc={personaDesc} />
  </div>
  );
}

function Persona({ isWizard, personaDesc }) {
  const persona_str = personaDesc.toString();
  const persona_style = "info-pane " + "persona-pane"
  if (persona_str === "") {
    return <div className={persona_style}></div>;
  }

  const persona_lines = personaDesc.split("\n").map((s, i) => {return <li key={i}><b>{s}</b></li>})

  const header = (isWizard === true) ?
    "Your chat partner has the following personality and interests." :
    "You will assume the following character, with appropriate related interests.";

  return (<div className={persona_style}>
    <h3>{header}</h3>
    <ul>{persona_lines}</ul>
  </div>);
}

function TaskDescription({ isWizard }) {
  const className = "info-pane " + "instruction-pane";
  // Apprentice
  const apprentice_description = (<div className={className}>
    <h3>Have a conversation with your chat partner about your favorite topic.</h3>
    <p>
      In this task, you will have a conversation with a chat partner who has knowledge
      about many things, and access to lots of information.
      You will be assigned a persona;
      the purpose of the task is to then have an in-depth conversation about your assigned interests.
      Your partner will strive to enlighten you on these topics.
      Note that your conversational partner will not share any interests with you;
      the conversation should, and will, focus entirely on your assigned interests.
        </p>
  </div>);

  // Wizard
  const [isDetailsHidden, setIsDetailsHidden] = useState(true);
  const wizard_task_details = isDetailsHidden ? "" :
    (<div>
      <p>
        You can look up the information that you need during the conversation
        by searching the internet with the search bar provided here.
        The outcome of this search shows you a number of internet articles,
        separated into sentences.
        Try to use the information from these sentences to have an informed conversation.
        When you use the knowledge from one or more sentences,
        select those sentences before sending the message you crafted.
        Please conduct a natural conversation and avoid copy/paste.
      </p>
      <p>
        Your role in this conversation is to assist your partner in learning
        and discussing <strong>their</strong> interests in detail.
        Pretend that you are a knowledgeable entity with conversational ability;
        your personal interests do not matter for this conversation.
        At the end of the conversation,
        your partner should be happy to have talked with you,
        but should not know anything about you.
      </p>
    </div>);
  const triangleCharCode = isDetailsHidden ? 9658 : 9660;
  const hide_show_buttons_text = isDetailsHidden ? " more details" : " less details";

  const wizard_description = (<div className={className}>
    <h3>Have a conversation with your chat partner about their favorite topics.</h3>
    <p>
      You will have a conversation with a chat partner who is interested in a few topics.
      Your partner’s interests will be displayed to you ahead of time;
      the purpose of the conversation is to discuss your partner’s interests in detail.
    </p>
    <button className="expand-details-button" onClick={() => setIsDetailsHidden(!isDetailsHidden)}>
      <strong>{String.fromCharCode(triangleCharCode) + hide_show_buttons_text}</strong>
    </button>
    {wizard_task_details}
  </div>);

  return (isWizard === true) ? wizard_description : apprentice_description;
}