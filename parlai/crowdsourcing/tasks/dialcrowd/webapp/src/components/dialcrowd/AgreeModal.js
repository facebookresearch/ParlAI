/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React from "react";
import { Button, Modal, Form, Checkbox } from "antd";

/* eslint-disable no-unused-vars */
/* eslint-disable react/no-direct-mutation-state*/
/* eslint-disable react/jsx-key */

class ConsentFormInner extends React.Component {
  /* Consent form and checkboxes
   * Props
   * @{bool} initState: Whether the window is visible or not.
   * @{bool} forceShow: Whether the window is visible or not.
   * @{function} handleReject:
   * @{function} onAccept:
   * @{String} consent: PDF source of the consent form document.
   * @{Array} checkboxes: [{key: @{int}, content: @{string}}]
   */
  constructor(props) {
    super(props);

    let check = {};
    props.checkboxes.map(function(i) {
      check["checkbox[" + i.key + "]"] = false;
    });

    this.state = {
      visible: this.props.initState || true,
      page: 0,
      checked: check
    };
  }

  render() {
    const visible = this.state.visible || this.state.forceShow;

    const pages = this._makePages();
    if (pages.length == 0) {
      return null;
    } else {
      const page = pages[this.state.page];
      const footer =
        this.state.page === pages.length - 1
          ? this._showAcceptRejectButton(pages)
          : this._showPageButton();
      return (
        <>
          <style>
            {`
        .ant-modal-content {height: 100%; display: flex; flex-direction: column;}
            `}
          </style>

          <Modal
            visible={visible}
            title={page.title}
            width={page.width}
            height={page.height}
            closable={false}
            maskClosable={false}
            bodyStyle={{ flexGrow: 1 }}
            centered
            zIndex={1000}
            footer={footer}
          >
            {page.content}
          </Modal>
        </>
      );
    }
  }

  close = () => {
    this.setState({ visible: false });
  };

  handleReject = () => {
    Modal.warning({
      title: "Thank you for your time.",
      content: "Please return to Amazon Mechanical Turk",
      zIndex: 2000
    });
  };

  handleChange = e => {
    this.state.checked[e.target.id] = e.target.checked;
  };

  handleAccept = e => {
    let close = true;

    console.log(this.state);

    for (const [key, value] of Object.entries(this.state.checked)) {
      if (!value) {
        close = false;
      }
    }

    if (close) {
      this.close();
      if (this.props.onAccept !== undefined) {
        this.props.onAccept();
      }
    } else {
      this.handleReject();
      if (this.props.config) {
        this.close();
      }
    }
  };

  nextPage = () => {
    this.setState({ page: this.state.page + 1 });
  };

  prevPage = () => {
    this.setState({ page: this.state.page - 1 });
  };

  _showPageButton = () => {
    const handleReject = this.props.handleReject || this.handleReject;
    return (
      <>
        <Button key="reject" onClick={handleReject} danger>
          Reject
        </Button>
        <Button type="primary" onClick={this.nextPage}>
          Agree & Next
        </Button>
      </>
    );
  };

  _showAcceptRejectButton = pages => {
    const handleReject = this.props.handleReject || this.handleReject;
    const handleAccept = this.handleAccept;
    return (
      <>
        <Button
          key="accept"
          type="primary"
          closable="false"
          onClick={e => handleAccept(e)}
        >
          Start the task!
        </Button>
      </>
    );
  };

  _makePages = () => {
    let pages = [];

    // add page if consent form is given.
    if (this.props.consent !== undefined && this.props.consent !== "") {
      pages.push({
        width: "90%",
        height: "calc(100% - 20px)",
        title: <h2> Please read and agree to the consent form. </h2>,
        content: (
          <iframe src={this.props.consent} height="100%" width="100%"></iframe>
        )
      });
    }

    // add page if checkboxes are given.
    if (
      this.props.checkboxes !== undefined &&
      this.props.checkboxes.length > 0
    ) {
      pages.push({
        title: <h2>Please check all that apply.</h2>,
        content: this.props.checkboxes.map(checkbox => (
          <Form.Item
            name={`checkbox[${checkbox.key}]`}
            valuePropName="checked"
            rules={[
              {
                required: true,
                transform: value => value || undefined,
                type: "boolean",
                message:
                  "You must fulfill this requirement in order to do this task."
              }
            ]}
            onChange={e => this.handleChange(e)}
          >
            <Checkbox key={checkbox.key}>{checkbox.content}</Checkbox>
          </Form.Item>
        ))
      });
    }

    return pages;
  };
}

const ConsentForm = ConsentFormInner;

/* eslint-enable no-unused-vars */
/* eslint-enable react/no-direct-mutation-state*/
/* eslint-enable react/jsx-key */

export { ConsentForm };
