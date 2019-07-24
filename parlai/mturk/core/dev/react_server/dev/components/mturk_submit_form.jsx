/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from 'react';
import $ from 'jquery';
import 'fetch';

// If we're in the amazon turk HIT page (within an iFrame) return True
function inMTurkHITPage() {
  try {
    return window.self !== window.top;
  } catch (e) {
    return true;
  }
}

function postData(url = ``, data = {}) {
  return fetch(url, {
    method: 'POST',
    mode: 'cors',
    cache: 'no-cache',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
    },
    redirect: 'follow',
    referrer: 'no-referrer',
    body: JSON.stringify(data),
  });
}

// Callback for static submission
function allDoneCallback(sender_id, assign_id, worker_id, response_data) {
  if (inMTurkHITPage()) {
    let server_url = window.location.origin;
    let post_data = {
      assignment_id: assign_id,
      agent_id: sender_id,
      worker_id: worker_id,
      response_data: response_data,
      task_group_id: TASK_GROUP_ID,
    };
    // TODO We allow workers to submit even if our server goes down.
    // reconcile this data with the fact that we'll likely mark as an
    // abandon on our end and will want to query the data from amazon instead
    postData(server_url + '/submit_hit', post_data).then(
      res => { $('input#mturk_submit_button').click(); }
    );
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
    let response_data_input = null;
    if (this.props.response_data !== undefined) {
      response_data_input = <input
        id="responseData"
        name="responseData"
        value={JSON.stringify(this.props.response_data)}
        readOnly
      />
    }
    return (
      <form
        id="mturk_submit_form"
        action={this.props.mturk_submit_url}
        method="post"
        style={{ display: 'none' }}
      >
        <input
          id="assignmentId"
          name="assignmentId"
          value={this.props.assignment_id}
          readOnly
        />
        <input id="hitId" name="hitId" value={this.props.hit_id} readOnly />
        <input
          id="workerId"
          name="workerId"
          value={this.props.worker_id}
          readOnly
        />
        {response_data_input}
        <input
          type="submit"
          value="Submit"
          name="submitButton"
          id="mturk_submit_button"
        />
      </form>
    );
  }
}

export { allDoneCallback, MTurkSubmitForm };
