/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { ErrorBoundary } from './error_boundary.jsx';
import { Checkboxes } from './checkboxes.jsx';
const ONBOARDING_MIN_CORRECT = 4;
const ONBOARDING_MAX_INCORRECT = 1;
const ONBOARDING_MAX_FAILURES_ALLOWED = 2;
var onboardingFailuresCount = 0;

var renderOnboardingFail = function () {
    // Update the UI
    document.getElementById("onboarding-submit-button").style.display = 'none';

    alert('Sorry, you\'ve exceeded the maximum amount of tries to label the sample conversation correctly, and thus we don\'t believe you can complete the task correctly. Please return the HIT.')
}

var handleOnboardingSubmit = function ({ onboardingData, annotationBuckets, yesNoBuckets, onSubmit }) {
    // OVERRIDE: Re-implement this to change onboarding success criteria
    console.log('handleOnboardingSubmit');
    var countCorrect = 0;
    var countIncorrect = 0;
    var onboardingAutoFail = false;

    var responses = [];
    for (var turnIdx = 0; turnIdx < onboardingData.dialog.length; turnIdx++) {
        var modelUtteranceForTurn = onboardingData.dialog[turnIdx][1];
        var answersForTurn = modelUtteranceForTurn.answers;
        var mustPass = modelUtteranceForTurn.must_pass; // if true, this question *must* be correct for the user to pass the onboarding
        var responsesForTurn = []
        if (!answersForTurn) {
            continue
        } else {
            var utteranceIdx = turnIdx * 2 + 1;
            console.log('Checking answers for turn: ' + utteranceIdx);
            if (turnIdx == 0) {
                var checkboxStubNames = Object.keys(yesNoBuckets.config);
            } else {
                var checkboxStubNames = Object.keys(annotationBuckets.config);
            }
            for (var j = 0; j < checkboxStubNames.length; j++) {
                var c = checkboxStubNames[j];
                var checkbox = document.getElementById(c + '_' + utteranceIdx);
                if (checkbox.checked) {
                    responsesForTurn.push([turnIdx, utteranceIdx, j, c, 'checked'])
                    if (answersForTurn.indexOf(c) > -1) {
                        countCorrect += 1
                    } else {
                        countIncorrect += 1
                        if (mustPass) {
                            onboardingAutoFail = true;
                        }
                    }
                // if radio then we only check if the single correct answer checked no need for the unchecked
                } else if (annotationBuckets.type != "radio") {
                    responsesForTurn.push([turnIdx, utteranceIdx, j, c, 'unchecked'])
                    if (answersForTurn.indexOf(c) > - 1) {
                        countIncorrect += 1
                        if (mustPass) {
                            onboardingAutoFail = true;
                        }
                    }
                    // If not checked *and* not an answer 
                    // don't increment anything
                }
            }
        }
        responses.push(responsesForTurn)
    }
    console.log('correct: ' + countCorrect + ', incorrect: ' + countIncorrect);
    if (countCorrect >= ONBOARDING_MIN_CORRECT && countIncorrect <= ONBOARDING_MAX_INCORRECT && onboardingAutoFail == false) {
        onSubmit({ success: true, responses: responses });
    } else {
        if (onboardingFailuresCount < ONBOARDING_MAX_FAILURES_ALLOWED) {
            onboardingFailuresCount += 1;
            alert('You did not label the sample conversation well enough. Please try one more time!');
        } else {
            renderOnboardingFail();
            onSubmit({ success: false, responses: responses })
        }
    }
}

function OnboardingDirections({ children }) {
    return (
        <section className="hero is-light">
            <div className="hero-head" style={{ textAlign: 'center', width: '850px', padding: '20px', margin: '0px auto' }}>
                {children}
            </div>
        </section>
    );
}

function OnboardingUtterance({ annotationBuckets, annotationQuestion, turnIdx, text }) {
    var extraElements = '';
    if (turnIdx % 2 == 1) {
        extraElements = '';
        extraElements = (<span key={'extra_' + turnIdx}><br /><br />
            <span style={{ fontStyle: 'italic' }}><span dangerouslySetInnerHTML={{ __html: annotationQuestion }}></span><br />
                <Checkboxes annotationBuckets={annotationBuckets} turnIdx={turnIdx} askReason={false} />
            </span>
        </span>)
    }
    return (
        <div className={`alert ${turnIdx % 2 == 0 ? "alert-info" : "alert-warning"}`} style={{ float: 'left', display: 'table' }}>
            <span className="onboarding-text"><b>{turnIdx % 2 == 0 ? 'TEXT' : 'QUESTION(S)'}:</b> {text}
                <ErrorBoundary>
                    {extraElements}
                </ErrorBoundary>
            </span>
        </div>
    )
}

function OnboardingComponent({ onboardingData, annotationBuckets, annotationQuestion, onSubmit }) {
    let yesNoBuckets = Object.assign({}, annotationBuckets);
    yesNoBuckets.config = {
        yes: {
            name: 'Yes',
            description: 'Yes'
        },
        no: {
            name: 'No',
            description: 'No'
        }
    };
    return (
        <div id="onboarding-main-pane">
            <OnboardingDirections>
                <h3>Task Description</h3>
                <div>
                    To first learn about the labeling task, please evaluate the sentences below, choosing the correct option for each one.</div>
            </OnboardingDirections>
            <div style={{ width: '850px', margin: '0px auto', clear: 'both' }}>
                <ErrorBoundary>
                    <div>
                        {
                            onboardingData.dialog.map((turn, idx) => (
                                <div key={'turn_pair_' + idx}>
                                    <OnboardingUtterance
                                        key={idx * 2}
                                        annotationBuckets={annotationBuckets}
                                        annotationQuestion={annotationQuestion}
                                        turnIdx={idx * 2}
                                        text={turn[0].text} />
                                    <OnboardingUtterance
                                        key={idx * 2 + 1}
                                        annotationBuckets={idx == 0 ? yesNoBuckets : annotationBuckets}
                                        annotationQuestion={annotationQuestion}
                                        turnIdx={idx * 2 + 1}
                                        text={turn[1].text} />
                                </div>
                            ))
                        }
                    </div>
                </ErrorBoundary>
                <div style={{ clear: 'both' }}></div>
            </div>
            <hr />
            <div style={{ textAlign: 'center' }}>
                <button id="onboarding-submit-button"
                    className="button is-link btn-lg"
                    onClick={() => handleOnboardingSubmit({ onboardingData, annotationBuckets, yesNoBuckets, onSubmit })}
                >
                    Submit Answers
                </button>
            </div>
        </div>
    );
}

export { OnboardingComponent, OnboardingUtterance };