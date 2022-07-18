/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";
import {Button, Collapse, Form, Modal, Radio, Select, Table, Tooltip} from 'antd';
import {QuestionCircleOutlined} from '@ant-design/icons';
import {ConsentForm} from "./AgreeModal";
import {Markdown} from "./QuestionList.js"
import {showFeedbackQuestion} from "./QuestionList.js";
import {getStyle} from "./style.js";
import track, { useTracking } from 'react-tracking';
import TrackingSensor from './TrackingSensor.js';

const FormItem = Form.Item;
const Option = Select.Option;
const RadioGroup = Radio.Group;
const Panel = Collapse.Panel;

class WorkerCategory extends React.Component {
  /* Consent form and checkboxes
   * Props
   * @{function} onSubmit
   * @{Array} taskConfig: [{}, {}], content from config.json
   * @{Array} taskData: [sentence1, sentence2, ...]
   * @{function} tracking
   */
  constructor(props) {
    super(props);
    this.state = {
      current: 1,
      activeKey: ["1", "2"],
      flag: false,
      sent: []
    }
    this.annotations = [];
  }

  stateSet = (label, sent, i, id) => {
    let dic = {};
    let time = Date.now();

    dic["timestamp"] = time;
    dic["label"] = label;
    dic["sentence"] = sent;
    dic["userId"] = this.state.userID;
    dic["taskId"] = this.state.ID;
    dic["sentId"] = id
    
    if (this.state.sent.some(e => e.sentId == id)){
      for (var a = 0; a < this.state.sent.length; a++){
        if (this.state.sent[a]["sentId"] == id){
          this.state.sent[a]["duration"] = this.state.sent[a]["duration"] + time - this.state.prev;
        }
      }
    }
    else{
      dic["duration"] = dic["timestamp"] - this.state.prev;
      if (!sent){
        this.setState({sent: [dic]});
      }
      else{
        this.state.sent.push(dic);
      }
      
    }

    this.setState({prev: time});
  }

  pushAnnotations = (values) => {    
    /* Extract annotations from the form and push them to this.annotations. */
    for (const [key, value] of Object.entries(values)) {
      if (key === "FeedbackOpen|||1") {
        this.feedback = value;
      } else {
        this.annotations.push({
          id: key,
          value: value
        })
      }
    }
  }

  onFinish = values => {
    /* Format annotations on the form after submission */
    const nPage = Math.ceil(this.props.taskData.length / this.props.taskConfig.numofsent);

    /* Form items follow the following format:
    * answer[0]["duration"]
    * answers[0]["timestamp"]
    */
    for (const [key, value] of Object.entries(this.state.sent)){
      values[`answer[${value['sentId']}]["duration"]`] = value['duration']
      values[`answer[${value['sentId']}]["timestamp"]`] = value['timestamp']
    }

    this.pushAnnotations(values);
    if (this.state.current < nPage) {
      this.setState({current: this.state.current + 1});
      this.props.tracking.trackEvent({ action: 'next-page' });
    } else {
      this.setState({submitted: true});
      this.annotations.push({feedback: this.feedback});
      // send the annotation to the backend
      this.props.onSubmit(this.annotations);
    }
    
  }

  changeTab = activeKey => {
    if (activeKey.includes("3")){
      this.setState({
        prev: Date.now(),
        activeKey: activeKey
      })
    }
  }

  openInstructions = () => {
    if (!(this.state.activeKey.includes("2"))){
      let x = this.state.activeKey;
      x.push("2");
      this.setState({
        activeKey: x
      })
    }
  }

  render() {
    const {current} = this.state;
    const formItemLayout2 = {
      labelCol: {span: 14},
      wrapperCol: {span: 10},
      fontSize: 18,
      colon: false,
      labelAlign: 'left'
    };
    const formItemLayout = {
      wrapperCol: {span: 10},
      fontSize: 18,
      colon: false,
    }
    const radioStyle = {
      display: 'block',
      height: '30px',
      lineHeight: '30px',
    };

    let hasInstr = false;
    for (let question of this.props.taskConfig.questionCategories) {
      hasInstr = hasInstr || (question.instruction || '-') != '-';
    }

    let hasExample = false;
    for (let question of this.props.taskConfig.questionCategories) {
      hasExample = hasExample || (question.examples || []).length > 0;
    }
    
    let hasCtrExample = false;
    for (let question of this.props.taskConfig.questionCategories) {
      hasCtrExample = hasCtrExample || (question.counterexamples || []).length > 0;
    }
    
    const likerts = ['1 Strongly Disagree', '', '', '', '5 Strongly Agree'];
    const labels = this.props.taskConfig.questionCategories.map((c) => (c.title));
    let styles = getStyle(this.props.taskConfig.style);
    const nPage = Math.ceil(this.props.taskData.length / this.props.taskConfig.numofsent);
    
    let initialVal = {};

    // form initialization for consent
    if (this.props.taskConfig.requirements !== undefined 
        && this.props.taskConfig.requirements.length > 0){
          this.props.taskConfig.requirements.map((checkbox) => (
            initialVal[`checkbox[${checkbox.key}]`] = false
          ))
    }

    // form initialization for workers
    // `[${i}]["confidence"]`
    // `[${i}]["answer"]`
    if (this.props.taskData !== undefined && this.props.taskData.length > 0){
      this.props.taskData.map((sent, i) => {
        initialVal[`answer[${i}]["answer"]`] = null
        initialVal[`answer[${i}]["confidence"]`] = null
      })
    }

    return <div style={styles.global}>
      <div id="surveyContainer"></div>
      <Form onFinish={this.onFinish} style={{"marginBottom": 0.6}} initialValues={initialVal}>
        <Collapse defaultActiveKey={['1', '2']} onChange={this.changeTab} activeKey={this.state.activeKey}>
          {(this.props.taskConfig.generic_introduction || '-') == '-' ? null :
          <Panel header="Background" key="1"  style={styles.tabTitle}>
            <p style={styles.background}>
              <Markdown enableMarkdown={this.props.taskConfig.enableMarkdown}>
                {this.props.taskConfig.generic_introduction}
              </Markdown>
            </p>
            <ConsentForm consent={this.props.taskConfig.consent[1]} checkboxes={this.props.taskConfig.requirements}/>
          </Panel>
          }
          <Panel header="Instructions" key="2" style={styles.tabTitle}>
            <div style={styles.instruction}>
              {this.props.taskConfig.enableMarkdown ?
              <Markdown enableMarkdown={this.props.taskConfig.enableMarkdown}>
                {this.props.taskConfig.generic_instructions}
              </Markdown> : <b>{this.props.taskConfig.generic_instructions}</b>
              }
            </div>
            <p style={{
              ...styles.instruction,
              marginTop: '0px',
              marginBottom: '0px',
              fontSize: styles.instruction.fontSize - 2
            }}>
              We expect this HIT will take <b>{this.props.taskConfig.time} minute(s)</b> and we will pay <b>${this.props.taskConfig.payment}</b>.
            </p>
            <div style={{...styles.example, "fontSize": styles.example.fontSize + 4}}>
              <p><b>Categories</b></p>
            </div>

            <CatDescriptionTable questions={this.props.taskConfig.questionCategories} styles={styles} />
          </Panel>
          <Panel header="Start the task" key="3" style={styles.tabTitle} onClick={() => this.props.tracking.trackEvent({ action: 'click' })}>
            <div title="category classification">
              <div style={{"textAlign": "center"}}>
                <p style={{"textAlign": "center", "fontSize": 18,
                          marginTop: styles.global.spacing,
                          marginBottom: 10}}><b>"Select a category for the given text"</b></p>
              </div>
              {this.props.taskData.slice(
                (this.state.current-1) * this.props.taskConfig.numofsent,
                (this.state.current) * this.props.taskConfig.numofsent
              ).map((item, i) => (
                <div style={{"backgroundColor": "#f7f7f7", marginBottom: '20px'}}>
                  <TrackingSensor name={`sentence`} event={{order: i, sentId: item.id}}>
                    <FormItem 
                      className={'two-rows-label'}
                      {...formItemLayout2}
                      rules={[{required: true, message: 'Please select a category'}]}
                      name={`answer[${item.id}]["answer"]`}
                      label={<div style={{
                        "display": "inline-block",
                        "float": "left",
                        "whiteSpace": "normal",
                        "marginRight": "12px",
                        "text-align": "left",
                        "lineHeight": "15px",
                      }}
                        ><table style={{fontSize: styles.utterance.fontSize}} onShow={() => console.log('Show')}>
                          { item.sentences.map(
                              x =>
                                <tr style={{"display": "block"}}>
                                  <td style={{textAlign: 'right', verticalAlign: 'top', lineHeight: 1.5, color: 'rgb(40, 40, 40)'}}>
                                    {(/(^User:|^ *System:)/.exec(x) || [null])[0]}&nbsp;
                                  </td>
                                  <td style={{textAlign: 'left', verticalAlign: 'top', lineHeight: 1.5, color: 'rgb(40, 40, 40)'}}>
                                    {x.replace(/(^User: |^ *System: )/, '')}
                                  </td>
                                  
                                </tr>
                          )
                          } </table>                           
                        </div>}
                        hasFeedback
                      >
                        <Select 
                          onChange={(e) => {
                          this.stateSet(e, item.sentences, i, item.id);
                          this.props.tracking.trackEvent({ action: 'select-answer', answer: e, sentId: item.id, order: item.sentences })
                        } }
                          placeholder="Please select a category"
                          >
                          {labels.map((label, j) => (
                            <Option value={label}>{label}</Option>
                          ))}
                      </Select> 
                    </FormItem>
                  </TrackingSensor>
                  <div style={{marginBottom: "20px", marginLeft: "58.33%"}}>
                    {hasInstr ?
                        <InstructionModal questions={this.props.taskConfig.questionCategories} styles={styles} type='description' /> : null
                        }
                        {(hasExample || hasCtrExample) ?
                        <InstructionModal questions={this.props.taskConfig.questionCategories} styles={styles} type='example' /> : null
                    }
                  </div>
                  <FormItem
                    {...formItemLayout2}
                    label={<div style={{
                      color: "forestgreen",
                      "display": "inline-flex",
                      fontSize: 16
                    }}>
                    {"Confidence of your answer? (1: Not confident, 3: Very confident) "}
                      </div>}
                    name={`answer[${item.id}]["confidence"]`}
                    rules={[{required: true, message: 'Please select a confidence level'}]}
                  >
                    <RadioGroup onChange={
                      (e) => this.props.tracking.trackEvent({
                        action: 'select-confidence', answer: e.target.value, sentId: item.id, order: i 
                      }) }>
                        <Radio value="1">1</Radio>
                        <Radio value="2">2</Radio>
                        <Radio value="3">3</Radio>
                      </RadioGroup>

                  </FormItem>
                </div>
              ))
              }              

            </div>

            <div style={{"backgroundColor": "#C1E7F8"}}>
              <FormItem style={{"textAlign": "center"}}
                        wrapperCol={{span: 12, offset: 6}}
    >
    <p style={{"textAlign": "center", "fontSize": 15, "color": "black"}}>You will get the code after you complete the HIT.</p>
    {current < nPage ?
              <Button type="primary" htmlType="submit">
                Next {current}/{nPage}
              </Button>
    :
              <div title={"Feedback"} style={{margin: "10px", padding:24 }} >
                {showFeedbackQuestion(this.props.taskConfig.hasFeedbackQuestion)}
                <Button type="primary" htmlType="submit"  disabled={this.state.submitted}>Submit</Button>
              </div>
    }

    {this.state.submitted ?
    <Modal title="Thank you for your submission!" footer={null} visible={true} closable={false} /> : null
    }
                    </FormItem>
    </div>
          </Panel>
        </Collapse>

      </Form>
    </div>
  }
}



class _InstructionModal extends React.Component {
    /* Props
     * @{bool} initState: Whether the window is visible or not.
     * @{string} type: type of the content
     * questions
     * styles
     */
    constructor (props) {
    super(props)
    this.state = {
      visible: this.props.initState || false
    };
  }
  
  render() {
    const visible = this.state.visible || this.state.forceShow;
    const checkboxes = this.props.handleAccept || this.close;

    const footer = (
      <Button key="accept" type="primary"
              closable="false" onClick={this.close}>
        Close
      </Button>
    );

    const contentType = this.props.type;


    return (
      <>
        <span style={{paddingRight: '1em', fontSize: 16, color: '#1890ff'}}>
          <a onClick={ () => {
            this.setState({visible: true});
            this.props.tracking.trackEvent({ action: `open-modal-${contentType}` })
          } }>
            <QuestionCircleOutlined />
      &nbsp;{contentType == 'example' ? 'show examples' : 'show instructions'}
          </a>
        </span>
        
        <style>
          {`
        .ant-modal-content {height: 100%; display: flex; flex-direction: column;}
          `}
        </style>

        <Modal
          visible={visible}
          title="Instructions"
          width="80%"
          closable={true}
          maskClosable={false}
          bodyStyle={{flexGrow: 1}}
          centered
          zIndex={1000}
          footer={footer}
          onCancel={this.close}
        >
          {this.renderInstruction()}
        </Modal>
      </>
    );
  }

  renderInstruction () {
    const styleInst = {
      color: 'black',
      fontSize: 18
    };
    const styleExp = {
      color: 'black',
      fontSize: 16
    };

    return <>
      <div style={styleExp} >
        <CatDescriptionTable
          questions={this.props.questions}
          styles={this.props.styles}
          type={this.props.type}
        />
      </div>
    </>
}
close = () => {
    this.setState({visible: false});
    const contentType = this.props.type;
    this.props.tracking.trackEvent({ action: `close-modal-${contentType}` })

  }

}


const InstructionModal = track({
  page: 'WorkerCategoryModal',
})(_InstructionModal);


function CatDescriptionTable (props) {
  let columns_example = [{
    title: 'Category',
    dataIndex: 'title',
    key: 'title',
  }];

  let hasInstr = false;
  for (let question of props.questions) {
    hasInstr = hasInstr || (question.instruction || '-') != '-';
  }

  let hasExample = false;
  for (let question of props.questions) {
    hasExample = hasExample || (question.examples || []).length > 0;
  }
  let hasCtrExample = false;
  for (let question of props.questions) {
    hasCtrExample = hasCtrExample || (question.counterexamples || []).length > 0;
  }

  if (props.type != 'example' && hasInstr) {
    columns_example.push(
      {
        title: 'Instructions',
        dataIndex: 'instruction',
        key: 'instruction',
        width: (props.type == 'description' || (!hasExample && !hasCtrExample))
          ? "80%" : "20%",
        render: (instruction, record, renderIndex) => (
          <TrackingSensor name={'description'} event={{title: record.title}}>
            <span>{instruction}</span>
          </TrackingSensor>
        )
      }
    );
  }

  if (hasExample && props.type != 'description') {
    columns_example.push({
      title: 'Examples',
      dataIndex: 'examples',
      key: 'examples',
      render: _renderExamples,
      width: (props.type == 'example' || !hasInstr) ? "45%" : "35%"
    });
  }

  if (hasCtrExample && props.type != 'description') {
    columns_example.push({
      title: 'Counterexamples',
      dataIndex: 'counterexamples',
      key: 'counterexamples',
      render: _renderExamples,
      width: (props.type == 'example' || !hasInstr) ? "45%" : "35%"
    });
  }

  const { Track, trackEvent } = useTracking();
  return <Table rowKey="sentid"
                dataSource={props.questions}
                columns={columns_example} size="small"
                style={{...props.styles.example,
                        marginTop: `${props.styles.global.spacing}px`,
                        marginBottom: `${props.styles.global.spacing}px`}}
  />;
}


function _renderExamples(examples, record, renderIndex) {
  return <TrackingSensor name={`example`} event={{title: record.title}}>
    <ul>
      {(examples || []).map((example, index) => (
        <li key={index}>
          <table style={{marginBottom: "0.5em"}}>
            {example.content.split('\n').map(
              (line, i) => (
                /(^User:|^System:)/.exec(line) ?
                <tr>
                  <td style={{textAlign: 'left', verticalAlign: 'top', lineHeight: 1.5, color: 'rgb(40, 40, 40)', width: "4em", margin: 0}}>
                    {(/(^User:|^System:)/.exec(line) || [null])[0]}&nbsp;
                  </td>
                  <td style={{textAlign: 'left', verticalAlign: 'top', lineHeight: 1.5, color: 'rgb(40, 40, 40)', margin: 0}}>
                    {line.replace(/(^User: |^System: )/, '')}

                    {(example.explain !== undefined && example.explain !== ''
                    && i + 1 == example.content.split('\n').length) ?
                     <Tooltip title=<TrackingSensor name={`explain-${record.title}-${index}`} event={{content: example.explain}}><span>{example.explain}</span></TrackingSensor>>
                    &nbsp; <sub><a>because...</a></sub>
                     </Tooltip> : null}
                  </td>
                </tr>
                : <>
                  {line}
                  {(example.explain !== undefined && example.explain !== ''
                  && i + 1 == example.content.split('\n').length) ?
                   <Tooltip title={example.explain}>
                  &nbsp; <sub><a>because...</a></sub>
                   </Tooltip> : null}
                </>
              )
            )}
          </table>
        </li>
      ))}
    </ul>
  </TrackingSensor>;
}


function dispatchFn(data) {
  data = {
    ...data,
    time: Date(),
    timestamp: Date.now()
  };
  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push(data);
}


export default track({
  page: 'WorkerCategory',
}, {
  dispatch: dispatchFn
})(WorkerCategory);
