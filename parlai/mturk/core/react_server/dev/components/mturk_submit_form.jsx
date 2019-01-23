/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import $ from 'jquery';

// If we're in the amazon turk HIT page (within an iFrame) return True
function inMTurkHITPage() {
  try {
    return window.self !== window.top;
  } catch (e) {
    return true;
  }
}

// Callback for submission
function allDoneCallback() {
  if (inMTurkHITPage()) {
    $("input#mturk_submit_button").click();
  }
}

class MTurkSubmitForm extends React.Component {
  /* Intentionally doesn't render anything, but prepares the form
  to submit data when the assignment is complete */
  shouldComponentUpdate(nextProps, nextState) {
    return (
      this.props.mturk_submit_url != nextProps.mturk_submit_url ||
      this.props.assignment_id != nextProps.assignment_id ||
      this.props.worker_id != nextProps.worker_id ||
      this.props.hit_id != nextProps.hit_id
    );
  }

  render() {
    return (
      <form
        id="mturk_submit_form" action={this.props.mturk_submit_url}
        method="post" style={{"display": "none"}}>
          <input
            id="assignmentId" name="assignmentId"
            value={this.props.assignment_id} readOnly />
          <input id="hitId" name="hitId" value={this.props.hit_id} readOnly />
          <input
            id="workerId" name="workerId"
            value={this.props.worker_id} readOnly />
          <input
            type="submit" value="Submit"
            name="submitButton" id="mturk_submit_button" />
      </form>
    );
  }
}

export {allDoneCallback, MTurkSubmitForm};
