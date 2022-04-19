/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { Button } from "react-bootstrap";
import { valid_search_query } from "./Moderator.js";

export default function SearchPanel({ mephistoContext,
    searchResults, selected, handleSelect, setSearchQuery, isWizard }) {
    const { sendLiveUpdate } = mephistoContext;
    const { agentState } = mephistoContext;
    const { taskConfig } = mephistoContext;

    function handleQuery(props) {
        const { query } = props;
        setSearchQuery(query);
        GetSearchResults(query, sendLiveUpdate);
    }

    const wizard_sidepane = (agentState.wants_act) ? <div>
        <NoDocumentSelected
            key="noOptionSelected"
            selected={selected}
            selectedChange={handleSelect} 
        />
        <SearchBar
            onSubmit={handleQuery}
            taskConfig={taskConfig}
        />
        <SearchResults
            search_results={searchResults}
            selected={selected}
            selectedChange={handleSelect}
        />
    </div> : <div><h3>Wait for response!</h3></div>;

    const apprentice_sidepane = <div>Enjoy the conversation!</div>;

    const sidepane = (isWizard === true) ? wizard_sidepane : apprentice_sidepane;

    return sidepane;
}

function SearchResults({ search_results, selected, selectedChange }) {
    function SearchDocsGenerator(doc, doc_index) {
        const { title } = doc;
        const shifted_row_index = doc_index + 1;
        const sel_sen = selected[shifted_row_index].slice();
        return (
            <div className="search-results">
                <SearchDoc
                    key={title + "-" + doc_index.toString()}
                    document={doc}
                    doc_id={shifted_row_index}
                    selected_sentences={sel_sen}
                    onChange={selectedChange}
                />
            </div>

        );
    }
    const elements = search_results.map(SearchDocsGenerator);
    return (<div>
        {elements}
    </div>);
}

function GetSearchResults(query, sendLiveUpdate) {
    const q = query;
    const ts = Date.now();
    const message = {
        timestamp: ts,
        episode_done: false,
        id: "Wizard",
        text: query,
        is_search_query: true
    };
    sendLiveUpdate(message);
}


function SearchBar({ onSubmit, taskConfig }) {
    const [text, setText] = useState("");

    function handleSubmit() {
        if (text === "") {
            alert("Please insert a term for search in the searh bar");
            return;
        }

        if (valid_search_query(text, taskConfig)) {
            onSubmit({ query: text });
        }
    }

    function handleKeyPresse(event) {
        if (event.key === "Enter") {
            handleSubmit();
        }
    }

    return (
        <div id="search-bar">
            Search web:
            <input
                type="text"
                value={text}
                onKeyPress={(event) => handleKeyPresse(event)}
                onChange={(event) => setText(event.target.value)}
            />
            <Button
                onClick={() => handleSubmit()}
            >
                Search
            </Button>
        </div>
    );
}

function SearcDocTitle({ title, opened, onOpenSelected }) {
    // choose the icon unicde (down or right pointing icon)
    const triangleCharCode = opened ? 9660 : 9658;
    const text = String.fromCharCode(triangleCharCode) + title;

    return (
        <div className="doc-title">
            <button className="doc-button" onClick={() => onOpenSelected()}>
                <strong>{text}</strong>
            </button>
        </div>
    );
}

function SearchDocSentence({ sentence, selected, onChange, loc }) {
    const sent_id = loc[1];
    const isChecked = selected[sent_id];
    return (
        <div className="doc-sentence">
            <input
                type="checkbox"
                checked={isChecked}
                onChange={() => onChange(loc)}
            />
            {sentence}
        </div>
    );
}

function SearchDoc({ document, doc_id, selected_sentences, onChange }) {
    const title = document.title;
    const sentences = document.content;
    const [opened, setOpened] = useState(false);

    function SentenceCheckBoxGenerator(sentence, sentenc_id) {
        const location = [doc_id, sentenc_id];
        const key = doc_id.toString() + "_" + sentenc_id.toString();
        return (
            <SearchDocSentence
                key={key}
                sentence={sentence}
                selected={selected_sentences}
                onChange={onChange}
                loc={location}
            />
        );
    }

    const sents = sentences.slice();
    const sentence_selectors = sents.map(SentenceCheckBoxGenerator);
    return (
        <div>
            <SearcDocTitle title={title}
                opened={opened}
                onOpenSelected={() => (setOpened(!opened))} />
            {(opened) ? sentence_selectors : null}
        </div>
    );
}

function NoDocumentSelected({ selected, selectedChange }) {
    if ((!selected) || (selected.length === 0)) {
        return null;
    }
    const loc = [0, 0];
    const isChecked = selected[0][0];
    return (
        <div id="no-answer">
            <input
                type="checkbox"
                checked={isChecked}
                onChange={() => selectedChange(loc)}
            />
            <span style={{ color: "blue" }}>
                Did not use search results for this message.
            </span>
        </div>
    )
}