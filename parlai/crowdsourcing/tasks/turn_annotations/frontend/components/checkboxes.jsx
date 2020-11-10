/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

function Checkboxes({ 
  annotationBuckets, 
  turnIdx, 
  askReason, 
  annotations, 
  onUpdateAnnotations, 
  enabled=true, 
}) {
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
  return (
    <div key={'checkboxes_' + turnIdx}>
      {
        Object.keys(annotationBuckets).map(c => (
          <span key={'span_' + c + '_' + turnIdx}>
            <input 
              type={input_type}
              id={c + '_' + turnIdx} 
              name={'checkbox_group_' + turnIdx} 
              onChange={(evt) => {
                let newVal = evt.target.checked;
                let oldAnnotations = Object.assign({}, annotations);
                oldAnnotations[annotationBuckets[c].value] = newVal;
                onUpdateAnnotations(oldAnnotations);
              }} 
              disabled={!enabled}
            />
            <span style={{ marginRight: '15px' }}>
              {annotationBuckets[c].name}
            </span>
          </span>
        ))
      }
      <div id={'checkbox_description_' + turnIdx} style={{ height: '24px' }}></div>
      {reasonComponent}
    </div>
  )
}

export { Checkboxes };