/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

var handleCheckboxChange = function (evt, annotationBuckets, onUserInputUpdate) {
  var checkboxId = evt.target.id;
  var whichCheckbox = checkboxId.substring(0, checkboxId.lastIndexOf('_'));
  var turnIdx = checkboxId.substring(checkboxId.lastIndexOf('_') + 1);
  var reasonHtml = '';

  if (evt.target.checked) {
    let checkboxDict = annotationBuckets.config[whichCheckbox];
    var checkboxPrettyName = ('prettyName' in checkboxDict) ? checkboxDict['prettyName'] : checkboxDict['name'];
    var checkboxDescription = checkboxDict['description'];
    reasonHtml = 'You labeled this as <b>' + checkboxPrettyName + '</b>, meaning that: ' + checkboxDescription;
  }
  document.getElementById('checkbox_description_' + turnIdx).innerHTML = reasonHtml;
  if (onUserInputUpdate) {
    onUserInputUpdate();
  }
}

function Checkboxes({ annotationBuckets, turnIdx, onUserInputUpdate, askReason }) {
  var reasonComponent = (
    <div>
      <br></br>
      <div>
        <div>Why did you select the checkboxes you did?</div>
        <input type="text" id={'input_reason_' + turnIdx} style={{ minWidth: '50%' }} />
      </div>
    </div>
  )
  if (!askReason) {
    reasonComponent = '';
  }
  let input_type = annotationBuckets.type !== undefined ? annotationBuckets.type : "checkbox";
  const showLineBreaks = true;  // pass this in in annotationBuckets
  const numBuckets = Object.keys(annotationBuckets.config).length;
  return (
    <div key={'checkboxes_' + turnIdx}>
      {
        Object.keys(annotationBuckets.config).map((c, checkboxIdx) => (
          <>
            <span key={'span_' + c + '_' + turnIdx}>
              <input
                type={turnIdx == 0 ? "radio" : input_type}
                id={c + '_' + turnIdx}
                name={'checkbox_group_' + turnIdx}
                onChange={(evt) => handleCheckboxChange(evt, annotationBuckets, onUserInputUpdate)}
              />
              <span style={{ marginRight: '15px' }}>
                {annotationBuckets.config[c].name}
              </span>
            </span>
            {(showLineBreaks && checkboxIdx < numBuckets - 1) ? <br></br> : ''}
          </>
        ))
      }
      <div id={'checkbox_description_' + turnIdx} style={{ height: '24px' }}></div>
      {reasonComponent}
    </div>
  )
}
// showLineBreaks: show a line break after every checkbox other than the final one

export { Checkboxes };