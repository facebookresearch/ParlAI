/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import {FormControl, Button} from 'react-bootstrap';
import {getCorrectComponent} from './core_components.jsx';
import $ from 'jquery';

// Custom message component shows context if it exists:
class ImageCaptionPane extends React.Component {
  render() {
    let image_data = "";
    if (this.props.task_data && this.props.task_data.image) {
      let image = this.props.task_data.image;
      image_data = `data:image/jpeg;base64,${image}`;
    }
    return <div>
      <img src={image_data} id="comment-image" style={{maxWidth: '100%', maxHeight: '60%'}} />
    </div>;
  }
}

export default {
  XContentPane: {'default': ImageCaptionPane},
};
