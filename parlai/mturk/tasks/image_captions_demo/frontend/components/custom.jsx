/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE This frontend is terrible and reflects a work in progress more than
// something that should be replicated.

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';
import {getCorrectComponent} from './core_components.jsx';
import $ from 'jquery';

// Custom message component shows context if it exists:
class ImageCaptionPane extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      textval: '',
    }
  }

  handleKeyPress(e) {
    if (e.key === 'Enter') {
      e.stopPropagation();
      e.nativeEvent.stopImmediatePropagation();
    }
  }

  validateEntry(textval) {
    if (textval.split(' ').length >= this.props.task_data.word_min &&
        textval.split(' ').length <= this.props.task_data.word_max
    ) {
      this.props.onValidDataChange(true, { caption: textval });
    } else {
      this.props.onValidDataChange(false, {});
    }
    this.setState({textval: textval});
  }

  render() {
    let image_data = "";
    if (this.props.task_data && this.props.task_data.image) {
      let image = this.props.task_data.image;
      image_data = `data:image/jpeg;base64,${image}`;
    }
    let text_input = (
      <FormControl
        type="text"
        id="id_text_input"
        style={{
          width: '80%',
          height: '100%',
          float: 'left',
          fontSize: '16px',
        }}
        value={this.state.textval}
        placeholder="Enter caption here."
        onKeyPress={e => this.handleKeyPress(e)}
        onChange={e => this.validateEntry(e.target.value)}
      />
    );
    return <div style={{padding: '20px'}}>
      <img src={image_data} id="comment-image" style={{maxWidth: '100%', maxHeight: '60%'}} />
      <br />
      <p><b>Please caption this image using between {this.props.task_data.word_min} and {this.props.task_data.word_max} words.</b></p>
      <br />
      {text_input}
    </div>;
  }
}

export default {
  XContentPane: {'default': ImageCaptionPane},
};
