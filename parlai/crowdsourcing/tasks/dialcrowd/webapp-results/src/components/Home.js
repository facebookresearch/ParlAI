/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {dialogs: []}
  }

  render() {
    return <div>
      <h1> What is DialCrowd? </h1>
      <p>In order to make testing easier and more automated, a framework for connecting to a crowdsourcing entity such
        as (but not only) <a href="https://www.mturk.com/">Amazon Mechanical Turk (AMT)</a> and for rapidly creating and
        deploying tasks. </p>
      <p>In our experience, there are often a small set of standardized test scenarios (templates) that cover a large
        portion of the dialog tasks that are run on AMT. When we make those scenarios easy to create, testing is faster,
        and less burdensome. For some who are new to the area, the use of these scenarios guides them to accepted
        testing structures that they may not have been aware of.</p>
      <p>At the outset, the Task Creation Toolkit will use standard templates that are easy to flesh
        out. Later, when the toolkit is finished, the community will be invited to add more.</p>
      <p>The Toolkit will aid in other aspects of running tasks such as, in relevant cases, reminding the requester to
        post a consent form for explicit permission to use the data. We will ensure that the results are collected
        ethically and can be made available to the community with as few restrictions as possible that do not compromise
        a worker’s privacy.</p>
      <h1> People </h1>
      <p>Jessica Huynh,
        Carnegie Mellon University</p>
      <p>Ting-Rui Chiang,
        Carnegie Mellon University</p>
      <p><a href="http://www.cs.cmu.edu/~kyusongl/" target={"_blank"} rel="noopener noreferrer">Kyusong Lee</a>,
        Carnegie Mellon University</p>
      <p><a href="http://www.cs.cmu.edu/~awb/" target={"_blank"} rel="noopener noreferrer">Maxine Eskenazi</a>, Carnegie
        Mellon University </p>
      <p><a href="https://www.cs.cmu.edu/~jbigham/" target={"_blank"} rel="noopener noreferrer">Jeffrey Bigham</a>,
        Carnegie Mellon University </p>

      <h1> Publications </h1>
      <p>Kyusong Lee, Tiancheng Zhao, Alan W Black and Maxine Eskenazi <a href="http://aclweb.org/anthology/W18-5028"
                                                                          target="_blank" rel="noopener noreferrer">DialCrowd:
        A toolkit for easy
        dialog system assessment</a>. Proceedings of the International Conference on Spoken Dialogue Systems Technology
        (SIGDIAL 2018) [DEMO PAPER], July 2018, Melbourne
      </p>
      <p><a
          href={"https://scholar.googleusercontent.com/scholar.bib?q=info:G7D4V4hp5xkJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW2aMLYe-N7TdJioOXsEjSn0HCTsTlM8Y&scisf=4&ct=citation&cd=-1&hl=en"}
          target={"_blank"} rel="noopener noreferrer">[BibTeX]</a></p>
      <h1> Contact </h1>
      <p> jhuynh@andrew.cmu.edu </p>

    </div>
  }
}

export default Home
