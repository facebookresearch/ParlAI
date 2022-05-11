/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";
import { useTracking } from 'react-tracking';
import VisibilitySensor from 'react-visibility-sensor';


function TrackingSensor (props) {
  const { Track, trackEvent } = useTracking();
  return <VisibilitySensor scrollDelay={15000} onChange={(isVisible) => {
    trackEvent({action: isVisible ? 'appear' : 'disappear', name: props.name, ...props.event});
  }}>
    {props.children}
  </VisibilitySensor>
}


export default TrackingSensor;
