/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
import React from "react";
import "./styles.css";

export function TaskDescription({ context }) {
    return (
        <div>
            <h1>Context</h1>
            <PersonsaDescription personsa={context.personas} />
            <LocationDescription location={context.location} />
            <h1>Instruction</h1>
            <GeneralDescription />
            <ExtraInformation />
        </div>
    )
}

function LocationDescription({ location }) {
    if (location == null) {
        return <Generating />;
    }
    const loc_str = ("name" in location) ? <div>
        <h2>You are in {location.name}</h2><p>{location.description}</p>
    </div> : <Generating />;
    return (
        <div className="location">
            {loc_str}
        </div>
    )
}

function PersonsaDescription({ personsa }) {
    if (personsa == null) {
        return <Generating />;
    }

    function PersonaCards(personsa, personsa_index) {
        const caption = (personsa_index === 0) ? <h2>You are {personsa.name}</h2> : <h2>{personsa.name}</h2>;
        const special_style = (personsa_index === 0) ? "player" : "npc";
        const full_class = "persona " + special_style;
        const desc = <p>{personsa.persona}</p>
        return (
            <div className={full_class}>
                {caption}
                {desc}
            </div>

        );
    }
    const elements = personsa.map(PersonaCards);
    return (<div>
        {elements}
    </div>);
}

function Generating() {
    return (
        <div>
            "Generating ..."
        </div>
    )
}

function GeneralDescription() {
    return (
        <div>
            <p>
                In this task you will have a conversation with two other characters in a fantasy game setting.
                The other two characters are controlled by Artificial Intelligence (AI).
                You will all be given characters and a description of the setting of the conversation.
            </p>
            <h4>Chat</h4>
            <p>
                You should play your character, conversing as if you were your character in the provided setting.
                The program decides whose turn is next.
                When it's your turn, the message bar at the bottom of the right panel is activated and you can play your role.
                Otherwise wait for others to play and evaluate their responses.
            </p>
            <h4>Evaluate</h4>
            <p>
                After each message from the AI, you will be asked to evaluate the response for its attributes:
                <ul>
                    <li><b>Consistent</b>: Does the response 1) make sense in the context of the conversation; 2) make sense in and of itself?</li>
                    <li><b>Engaging</b>: Are you engaged by the response? Do you want to continue the conversation?</li>
                    <li><b>Out of turn</b>: It didn't make sense for that character to speak at that point?</li>
                    <li><b>Mistaken identity</b>: Does it speak like it is someone else?</li>
                    <li><b>Contradictory</b>: Contradicts the character's description of what it said before?</li>
                    <li><b>Nonsensical</b>: Doesn't make any sense in this context.</li>
                </ul>
                You must check at least one box per response, which can be “None” if no attributes apply to the response.
            </p>
        </div>
    )
}

function ExtraInformation() {
    return (
        <div>
            <h3>What do I talk about?</h3>
            <p>
                Anything, so long as you remain in character.
                If it would make sense for your character, you could try to learn about your partners, or talk about yourself,
                or the setting you have all been assigned to.
            </p>
            <br />
            <h3>When does the task end?</h3>
            <p>
                The conversation will continue for a total of 15 messages.
                After reaching that limit you will see the button that allows you to send the chat and submit the HIT.
            </p>

            <br />
            <h3>What NOT to do:</h3>
            <ul>
                <li><b>Be aware the conversations you have will be made public, so act as you would e.g. on a public social network like Twitter.</b></li>
                <li>Do not talk about the task itself, or about MTurk</li>
                <li>Avoid racism, sexism, hate speech, and other forms of inappropriate conversation. </li>
                <li>Avoid <b>real world</b> topics and locations, and instead remain in the medieval fantasy setting.</li>
                <li>
                    Don't direct into conversations where you pretend to take actions
                    (like "here you go! *gives item*"), stick to strictly chat.
                </li>
                <li>
                    Don't idle for too long (4 minutes) or you will be disconnected from the chat
                    and unable to submit.
                </li>
            </ul>
        </div>
    )
}