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
const DEFAULT_MIN_CORRECT = 4;
const DEFAULT_MAX_INCORRECT = 3;
const DEFAULT_MAX_FAILURES_ALLOWED = 1;
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
    const min_correct = onboardingData.hasOwnProperty("min_correct") ? onboardingData.min_correct : DEFAULT_MIN_CORRECT;
    const max_incorrect = onboardingData.hasOwnProperty("max_incorrect") ? onboardingData.max_incorrect : DEFAULT_MAX_INCORRECT;
    const max_failures_allowed = onboardingData.hasOwnProperty("max_failures_allowed") ? onboardingData.max_failures_allowed : DEFAULT_MAX_FAILURES_ALLOWED;
    if (countCorrect >= min_correct && countIncorrect <= max_incorrect) {
        onSubmit({ annotations: currentTurnAnnotations, success: true });
    } else {
        if (onboardingFailuresCount < max_failures_allowed) {
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
    text, 
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
    if (onboardingData === null) {
        return (
            <div id="onboarding-main-pane">
                Please wait while we set up the task...
            </div>
        );
    } else {
        const [currentTurnAnnotations, setCurrentAnnotations] = React.useState(
            Array.from(Array(onboardingData.dialog.length), () => Object.fromEntries(
                Object.keys(annotationBuckets.config).map(bucket => [bucket, false]))
            )
        );
        return (
            <div id="onboarding-main-pane">
                <OnboardingDirections>
                    <h1> Informed Consent For Research </h1>

                    <h3>Introduction</h3>
                    <div>
                        We invite you to take part in a research study. Please take as much time as you need to read the consent form. You may want to discuss it with your family, friends, or your personal doctor. If you find any of the language difficult to understand, please ask questions. If you decide to participate, you will be asked to sign this form. A copy of the signed form will be provided to you for your records.
                    </div>

                    <h3>Key Information</h3>
                    <div>
                        <p>
                            The following is a short summary of this study to help you decide whether you should participate. More detailed information is listed later in this form.
                        </p>

                        <ol>
                            <li>
                                Being in this research study is voluntary - it is your choice.
                            </li>
                            <li>
                                You are being asked to take part in this study because you are bilingual, speaking both  [TARGET LANGUAGE] and English fluently, and you are at least 18 years old. The purpose of this study is to assess the ability of artificial intelligence-based moderators ("bots") to affect online toxic behavior. Your participation in this study will last approximately 10 minutes per conversation you wish to participate in, but you will have multiple opportunities to participate. Procedures will include reading an existing partial online conversation, assuming the persona, attitude, opinion, and behavior of one of the participants in that conversation, and reacting naturally to another participant who tries to alter "your" behavior.  
                            </li>
                            <li>
                                There are risks from participating in this study. The most common risks are that you will feel uncomfortable imitating the personality of someone whose views or style are dissimilar from your own, or that you will feel reluctant to discuss the topics of the given conversation. More detailed information about the risks of this study can be found under the “Risk and Discomfort” section. 
                            </li>
                            <li>
                                You may not receive any direct benefit from taking part in this study. However, your participation in this study may help us learn how to create better bots and this can lead to less toxic behavior online.
                            </li>
                            <li>
                                If you decide not to participate in this research, your other choices may include finding other tasks on Mechanical Turk that you find of greater interest. 
                            </li>
                        </ol>
                    </div>
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
                                            annotations={currentTurnAnnotations[idx]}
                                            turnIdx={idx * 2}
                                            text={turn[0].text} />
                                        <OnboardingUtterance
                                            key={idx * 2 + 1}
                                            annotationBuckets={annotationBuckets}
                                            annotationQuestion={annotationQuestion}
                                            turnIdx={idx * 2 + 1}
                                            text={turn[1].text} 
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
                        Submit
                    </button>
                </div>
            </div>
        );
    }
}

export { OnboardingComponent, OnboardingUtterance };