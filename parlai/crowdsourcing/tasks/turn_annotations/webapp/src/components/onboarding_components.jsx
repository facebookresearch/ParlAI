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
const ONBOARDING_MAX_INCORRECT = 3;
const ONBOARDING_MAX_FAILURES_ALLOWED = 1;
var onboardingFailuresCount = 0;

var renderOnboardingFail = function () {
    // Update the UI
    document.getElementById("onboarding-submit-button").style.display = 'none';

    alert('Sorry, you\'ve exceeded the maximum amount of tries to label the sample conversation correctly, and thus we don\'t believe you can complete the task correctly. Please return the HIT.')
}

var handleOnboardingSubmit = function ({ onboardingData, annotationBuckets, onSubmit }) {
    // OVERRIDE: Re-implement this to change onboarding success criteria
    console.log('handleOnboardingSubmit');
    var countCorrect = 0;
    var countIncorrect = 0;
    for (var turnIdx = 0; turnIdx < onboardingData.dialog.length; turnIdx++) {
        var modelUtteranceForTurn = onboardingData.dialog[turnIdx][1];
        var answersForTurn = modelUtteranceForTurn.answers;
        if (!answersForTurn) {
            continue
        } else {
            var utteranceIdx = turnIdx * 2 + 1;
            console.log('Checking answers for turn: ' + utteranceIdx);
            var checkboxStubNames = Object.keys(annotationBuckets.config);
            for (var j = 0; j < checkboxStubNames.length; j++) {
                var c = checkboxStubNames[j];
                var checkbox = document.getElementById(c + '_' + utteranceIdx);
                if (checkbox.checked) {
                    if (answersForTurn.indexOf(c) > -1) {
                        countCorrect += 1
                    } else {
                        countIncorrect += 1
                    }
                } else {
                    if (answersForTurn.indexOf(c) > - 1) {
                        countIncorrect += 1
                    }
                    // If not checked *and* not an answer 
                    // don't increment anything
                }
            }
        }
    }
    console.log('correct: ' + countCorrect + ', incorrect: ' + countIncorrect);
    if (countCorrect >= ONBOARDING_MIN_CORRECT && countIncorrect <= ONBOARDING_MAX_INCORRECT) {
        onSubmit({ success: true });
    } else {
        if (onboardingFailuresCount < ONBOARDING_MAX_FAILURES_ALLOWED) {
            onboardingFailuresCount += 1;
            alert('You did not label the sample conversation well enough. Please try one more time!');
        } else {
            renderOnboardingFail();
            onSubmit({ success: false })
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
        <div className={`alert ${turnIdx % 2 == 0 ? "alert-info" : "alert-warning"}`} style={{ float: `${turnIdx % 2 == 0 ? "right" : "left"}`, display: 'table' }}>
            <span className="onboarding-text"><b>{turnIdx % 2 == 0 ? 'YOU' : 'THEM'}:</b> {text}
                <ErrorBoundary>
                    {extraElements}
                </ErrorBoundary>
            </span>
        </div>
    )
}

function OnboardingComponent({ onboardingData, annotationBuckets, annotationQuestion, onSubmit }) {
    return (
        <div id="onboarding-main-pane">
            <OnboardingDirections>
                <h3>Task Description</h3>
                <div>
                    To first learn about the labeling task, please evaluate the "THEM" speaker in the conversation below, choosing the correct checkboxes.</div>
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
                                        annotationBuckets={annotationBuckets}
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
                    onClick={() => handleOnboardingSubmit({ onboardingData, annotationBuckets, onSubmit })}
                >
                    Submit Answers
                </button>
            </div>
        </div>
    );
}

export { OnboardingComponent, OnboardingUtterance };