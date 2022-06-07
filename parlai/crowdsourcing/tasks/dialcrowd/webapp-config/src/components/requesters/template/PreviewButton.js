/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";
import {Button, Modal} from 'antd';


class PreviewButton extends React.Component {
  constructor (props) {
    super(props)
    this.state = {visible: this.props.initState || false};
  }
  
  render() {
    const {visible} = this.state;
    const formItemLayoutWithOutLabel = {
      wrapperCol: {
        xs: {span: 24, offset: 0},
        sm: {span: 20, offset: 4},
      },
    };
    return (
      <>
        <Button {...formItemLayoutWithOutLabel}
                disabled={this.props.consent === undefined || this.props.consent === ""}
                onClick={this.open}
                style={{width: '90%'}}>
          Preview Consent Form
        </Button>
        <style>
          {`
        .ant-modal-content {height: 100%; display: flex; flex-direction: column;}
          `}
        </style>
        <Modal
          visible={visible}
          title="Preview of Consent Form"
          width="80%"
          height="90%"
          closable={true}
          maskClosable={true}
          bodyStyle={{flexGrow: 1}}
          onCancel={this.close}
          centered
          footer={
            <Button key="Close" type="primary" onClick={this.close}>
              Close
            </Button>
          }
        >
          <iframe src={this.props.consent} width="100%" height="100%"></iframe>
        </Modal>
      </>
    );
  }
  
  open = () => {
    this.setState({visible: true});
  }
  
  close = () => {
    this.setState({visible: false});
  }

}


export default PreviewButton;
