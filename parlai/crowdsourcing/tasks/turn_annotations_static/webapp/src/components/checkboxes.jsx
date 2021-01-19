/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";

function CheckboxDescription({
  checkboxConfig,
  turnIdx,
}) {
  const id = 'checkbox_description_' + turnIdx;
  const style = {height: '24px'};
  if (checkboxConfig) {
    const checkboxPrettyName =
      checkboxConfig.prettyName ?? checkboxConfig.name;
    var checkboxDescription = checkboxConfig.description;
    return (
      <div id={id} style={style}>
        You labeled this as <b>{checkboxPrettyName}</b>, meaning that:{' '}
        {checkboxDescription}
      </div>
    );
  }
  return <div id={id} style={style}></div>;
}

// type CheckboxesProps = {
//   annotationBuckets: AnnotationBuckets,
//   turnIdx: number,
//   turnAnswers: TurnAnswers,
//   onUserInputUpdate?: () => void,
//   askReason: boolean,
//   lastSelected: ?string,
//   setLastSelected: (?string) => void,
//   setChecked: (number, string, boolean) => void,
//   setReason: (number, string) => void,
// };

class Checkboxes extends React.Component {
  constructor(props) {
    super(props);
  }

  handleCheckboxChange(evt, onUserInputUpdate) {
    var checkboxId = evt.target.id;
    var whichCheckbox = checkboxId.substring(0, checkboxId.lastIndexOf('_'));
    this.props.setChecked(
      this.props.turnIdx,
      whichCheckbox,
      evt.target.checked,
    );
    this.props.setLastSelected(evt.target.checked ? whichCheckbox : null);

    if (onUserInputUpdate) {
      onUserInputUpdate();
    }
  }

  handleReasonChange = evt => {
    this.props.setReason(this.props.turnIdx, evt.target.value);
  };

  render() {
    const {
      annotationBuckets,
      turnAnswers,
      turnIdx,
      onUserInputUpdate,
      askReason,
      lastSelected,
    } = this.props;
    var reasonComponent = (
      <div>
        <br></br>
        <div>
          <div>Why did you select the checkboxes you did?</div>
          <input
            type="text"
            id={'input_reason_' + turnIdx}
            style={{minWidth: '50%'}}
            value={turnAnswers.input_reason}
            onChange={this.handleReasonChange}
          />
        </div>
      </div>
    );
    if (!askReason) {
      reasonComponent = '';
    }
    const input_type =
      annotationBuckets.type !== undefined
        ? annotationBuckets.type
        : 'checkbox';
    return (
      <div key={'checkboxes_' + turnIdx}>
        {Object.keys(annotationBuckets.config).map(bucket => {
          const c = annotationBuckets.config[bucket];
          return (
            <span key={'span_' + bucket + '_' + turnIdx}>
              <input
                type={input_type}
                id={bucket + '_' + turnIdx}
                name={'checkbox_group_' + turnIdx}
                checked={turnAnswers.buckets[bucket]}
                onChange={evt =>
                  this.handleCheckboxChange(evt, onUserInputUpdate)
                }
              />
              <span style={{marginRight: '15px'}}>{c.name}</span>
            </span>
          );
        })}
        <CheckboxDescription
          checkboxConfig={
            lastSelected ? annotationBuckets.config[lastSelected] : null
          }
          turnIdx={turnIdx}
        />
        {reasonComponent}
      </div>
    );
  }
}

export { Checkboxes };