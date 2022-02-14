/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";

class Default extends React.Component {
  constructor(props) {
    super(props);
    this.state = {dialogs: []}
  }
  render() {
    return <div>
      <h1> What is DialCrowd </h1>
      <p>In order to make testing easier and more automated, a framework for connecting to a crowdsourcing entity such as (but not only) Amazon Mechanical Turk (AMT) and for rapidly creating and deploying tasks. </p>
      <p>In our experience, there are often a small set of standardized test scenarios (templates) that cover a large portion of the dialog tasks that are run on AMT. When we make those scenarios easy to create, testing is faster, and less burdensome. For some who are new to the area, the use of these scenarios guides them to accepted testing structures that they may not have been aware of.</p>
      <p>While developing a dialog system, researchers often need to run a small set of standardized tests. These will also be available. At the outset, the Task Creation Toolkit will use standard templates that are easy to flesh out. Later, when the toolkit is finished, the community will be invited to add more.</p>
      <p>The Toolkit will aid in other aspects of running tasks such as, in relevant cases, reminding the requester to post a consent form for explicit permission to use the data. We will ensure that the results are collected ethically and can be made available to the community with as few restrictions as possible that do not compromise a worker’s privacy.</p>
    </div>
  }
}

export default Default
