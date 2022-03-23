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

function arraysEqual(_arr1, _arr2) {
    if (!Array.isArray(_arr1) || ! Array.isArray(_arr2) || _arr1.length !== _arr2.length)
      return false;

    var arr1 = _arr1.concat().sort();
    var arr2 = _arr2.concat().sort();
    for (var i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i])
            return false;
    }
    return true;
}

var handleOnboardingSubmit = function ({ onboardingData, currentTurnAnnotations, onSubmit }) {
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
            let givenAnswers = currentTurnAnnotations[turnIdx];
            let answerArray = [];
            for (let arrayKey in givenAnswers) {
                if (givenAnswers[arrayKey]) {
                    answerArray.push(arrayKey);
                }
            }
            if (arraysEqual(answerArray, answersForTurn)) {
                countCorrect += 1;
            } else {
                countIncorrect += 1;
            }
        }
    }
    console.log('correct: ' + countCorrect + ', incorrect: ' + countIncorrect);
    if (countCorrect >= ONBOARDING_MIN_CORRECT && countIncorrect <= ONBOARDING_MAX_INCORRECT) {
        onSubmit({ annotations: currentTurnAnnotations, success: true });
    } else {
        if (onboardingFailuresCount < ONBOARDING_MAX_FAILURES_ALLOWED) {
            onboardingFailuresCount += 1;
            alert('You did not label the sample conversation well enough. Please try one more time!');
        } else {
            renderOnboardingFail();
            onSubmit({ annotations: currentTurnAnnotations, success: false })
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

function OnboardingUtterance({ 
    annotationBuckets, 
    annotationQuestion, 
    turnIdx, 
    text = null, 
    text_response_1 = null,
    text_response_2 = null,
    annotations = null, 
    onUpdateAnnotation = null,
}) {
    var extraElements = '';
    if (turnIdx % 2 == 1) {
        extraElements = '';
        extraElements = (<span key={'extra_' + turnIdx}><br /><br />
            <span style={{ fontStyle: 'italic' }}><span dangerouslySetInnerHTML={{ __html: annotationQuestion }}></span><br />
                <Checkboxes 
                    annotations={annotations} 
                    onUpdateAnnotations={onUpdateAnnotation} 
                    annotationBuckets={annotationBuckets} 
                    turnIdx={turnIdx} 
                    askReason={false} 
                />
            </span>
        </span>)

        return (
            <div className={`alert alert-warning`} style={{ float: `left`, display: 'table', clear: `both` }}>
                <div className="onboarding-text">
                    <b>Response 1:</b> {text_response_1}
                    <br />
                    <b>Response 2:</b> {text_response_2}
                    <ErrorBoundary>
                        {extraElements}
                    </ErrorBoundary>
                </div>
            </div>
        )
    }
    return (
        <div className={`alert alert-info`} style={{ float: `right`, display: 'table', clear: `both`}}>
            <span className="onboarding-text"><b>You:</b> {text}
                <ErrorBoundary>
                    {extraElements}
                </ErrorBoundary>
            </span>
        </div>
    )
}

function OnboardingComponent({ onboardingData, annotationBuckets, annotationQuestion, onSubmit }) {
    if (onboardingData === null) {
        return (
            <div id="onboarding-main-pane">
                Please wait while we set up the task...
            </div>
        );
    } else {
        const [currentTurnAnnotations, setCurrentAnnotations] = React.useState(
            Array.from(Array(onboardingData.dialog.length), () => Object.fromEntries(
                annotationBuckets.map(bucket => [bucket.value, false]))
            )
        );
        return (
            <div id="onboarding-main-pane">
                <OnboardingDirections>
                    <h3>Task Description</h3>
                    <div>
                        To first learn about the labeling task, please choose the correct checkbox for the given question (in italics) for this conversation.</div>
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
                                            text_response_1={turn[1].text_response_1}
                                            text_response_2={turn[1].text_response_2}
                                            annotations={currentTurnAnnotations[idx]}
                                            onUpdateAnnotation={
                                                (newAnnotations) => {
                                                    let updatedAnnotations = currentTurnAnnotations.slice()
                                                    updatedAnnotations[idx] = newAnnotations;
                                                    setCurrentAnnotations(updatedAnnotations);
                                                }
                                            }
                                            />
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
                        onClick={() => handleOnboardingSubmit({ 
                            onboardingData, 
                            currentTurnAnnotations, 
                            onSubmit,
                        })}
                    >
                        Submit Answers
                    </button>
                </div>
            </div>
        );
    }
}

export { OnboardingComponent, OnboardingUtterance };