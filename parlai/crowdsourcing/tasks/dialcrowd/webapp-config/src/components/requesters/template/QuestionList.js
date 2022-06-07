/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from "react";
import {Button, Form, Input, Radio, Tooltip, Divider, InputNumber} from 'antd';
import {MinusCircleOutlined, ExclamationCircleOutlined, PlusOutlined, QuestionCircleOutlined} from '@ant-design/icons';
import PreviewQuestion, {PreviewExample} from "./Preview.js";


function lists2Questions (queries, types, optionss,
                          exampless=undefined, counterexampless=undefined) {
  const _addKey = (xs) => (
    xs.filter((x) => (x !== null))
      .map((x, i) => ({key: i, content: x}))
  );

  let questions = [];
  for (let i = 0; i < queries.length; i += 1) {
    if (exampless !== undefined) {
      // Workaround
      if (queries[i] === null) {
        continue;
      }
      // Workaround: in case exampless[i] is not an array.
      let examples = Array.isArray(exampless[i]) ? exampless[i]: [exampless[i]];
      let counterexamples = (
        Array.isArray(exampless[i]) ? counterexampless[i]: [counterexampless[i]]
      );

      questions.push({
        "key": i,
        "title": queries[i],
        "type": types[i],
        "options": (          
          optionss[i] === null ? [{key: 0, content: ""}] : _addKey(optionss[i])
        ),
        "examples": _addKey(examples),
        "counterexamples": _addKey(counterexamples),
      });
    } else {
      questions.push({
        "key": i,
        "title": queries[i],
        "type": types[i],
        "options": (          
          optionss[i] === null ? [{key: 0, content: ""}] : _addKey(optionss[i])
        )
      });
    }
  }
  return questions;
}


function addKeys (xs) {
  if (Array.isArray(xs)) {
    xs = xs.filter((x) => (x !== null));
    if (typeof(xs[0]) === 'string') {
      return xs.map((x, i) => ({key: i, content: x}));      
    } else {
      return xs.map((x, i) => ({key: i, ...addKeys(x)}));
    }
  } else if (typeof(xs) === 'object') {
    let newX = {};
    for (let [k, x] of Object.entries(xs)) {
      newX[k] = addKeys(x);
    }
    return newX;
  } else {
    return xs;
  }
}


class QuestionList extends React.Component {
  /* Props:
   * @{object}	form:
   * @{style}	formItemLayout:
   * @{Array} questions [
   *    {
   *        'key': @{int},
   *        'title': @{string},
   *        'type': @{string},
   *        'options': [{'key': @{int} key, 'content': @{string}}, ... ]
   *        'examples': [{'key': @{int} key, 'content': @{string}}, ... ]
   *        'counterexamples': [{'key': @{int} key, 'content': @{string}}, ... ]
   *    }, ...
   * ]
   * @{string} questionFieldLabel: default `Question`.
   * @{Array} rootKey: 
   * @{Array} systemNames:  For the preview of A/B test questions.
   * @{string} questionHelpText: Like "After workers talk to your dialog systems,"
   * @{string} textAddQuestion: Text to show on the add question button.
   * @{string} textInstruction: Text to show in the "tip" tooltip.
   * "please provide a survey question on how your systems performed."
   * @{function} removeByKey: Function that can remove a element by id.
   * @{function} addByKey: Function that can remove a element by id.
   * @{function} updateByKey: Function that can update a element by id.
   * @{string} placeholderQuestion: 
   * @{string} placeholderExample: 
   * @{string} placeholderCounterexample: 
   * @{string} placeholderOption: 
   * @{string} listStyle: Either `box` or `divider`.
   * @{bool}  noPreview: No preview button.
   */
  static questionTypes = [
    ["Likert Scale", "Likert Scale",
     "Give the system a score from 1 to 5, 1 as strongly disagree and 5 as strongly agree."],
    ["Open ended", "Open ended",
     "Require more thought and more than a simple one-word answer."],
    ["Radio", "Multiple Choice",
     "Multiple choices are given."]
  ];

  constructor (props) {
    super(props);
    this.formItemLayout = props.formItemLayout;
    this.formItemLayoutWithOutLabel = {
      wrapperCol: {
        xs: {span: 24, offset: 0},
        sm: {span: 20, offset: 4},
      },
    };
    this.styleWarning = {
      'width': '90%', 'color': 'darkorange',
      'text-align': 'center',
      'font-weight': 'bold'
    };
  }

  render () {
    return (
      <>
        {this.props.questions.map(
          (question) => (this._renderQuestionSection(question))
        )}
        <Form.Item {...this.formItemLayoutWithOutLabel}>
          <Button type="dashed" onClick={this._addQuestion} style={{width: '90%'}}>
            <PlusOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} /> {this.props.textAddQuestion}
          </Button>
        </Form.Item>
        <Form.Item {...this.formItemLayoutWithOutLabel}>
          <PreviewExample questions={this.props.questions} />
        </Form.Item>
      </>
    );
  }

  static newQuestion = () => {
    const newQuestion = {
      "title": "",
      "type": "Open ended",
      "options": [{"key": 0, "content": ""}],
      "examples": [{"key": 0, "content": ""}],
      "counterexamples": [{"key": 0, "content": ""}],
      "likertScale": 5
    };
    return newQuestion;
  }
  
  _addQuestion = () => {
    this.props.addByKey(
      this.props.rootKey, this.constructor.newQuestion()
    );
  };
  
  _renderQuestionSection (question) {
    switch (this.props.listStyle || 'box') {
      case 'box': 
        return (
          <div key={question.key}
               style={{ border: "2px solid black", margin: "10px", padding: 24}} >
            {this._renderQuestionBody(question)}
          </div>
        );
      case 'divider': 
        return (
          <div key={question.key}>
            {this._renderQuestionBody(question)}
            <Divider />
          </div>
        );
      default: {
        throw `Unsupported listStyle ${this.props.listStyle}`;
      }
    }
  }

  _renderQuestionBody (question) {
    const fieldNameQuestion = (
      String(this.props.rootKey[0])
      + this.props.rootKey.slice(1).map((key) => (`[${key}]`)).join('')
    );
    const handleChange = (fieldName) => (event) => {
      const value = event.target !== undefined ? event.target.value : event;
      this.props.updateByKey(
        this.props.rootKey.concat([question.key]),
        {[fieldName]: value}
      )
    };
    return (
      <>
        {/* instruction tooltip that show instruction when mouse hovering over it. */}
        
        {this.props.textAddQuestion !== "Add a System Specific Question" ? 
          <span style={{float: "left"}}>
            <Tooltip
             title={this.props.textInstruction}>
                <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }}/> 
                <span style={{ display: 'inline-block', verticalAlign: 'middle' }}>&nbsp; Tips  </span>  
           </Tooltip> 
           </span> : null
        }

        {/* Remove question button. */}
        <Form.Item>
          <span style={{float: "right", "margin-bottom": "-30px", "margin-top": "-20px", "margin-right": "-10px"}}>
            {this.props.questions.length > 1 ? (
              <div
                disabled={this.props.questions.length === 1}
                onClick={() => this.props.removeByKey(
                  this.props.rootKey.concat([question.key])
                )}
                style={{ cursor: "pointer" }}
                >
                  <MinusCircleOutlined />
              </div>
            ) : null}
          </span>
        </Form.Item>

        {/* Type of question. */}
        {this._showQuestionType(question)}

        {/* Questions */}
        <Form.Item {...(this.formItemLayout)}
                   label={(
                     <span>
                       {this.props.questionFieldLabel || "Question"} &nbsp;
                       <Tooltip title={this.props.questionHelpText}>
                        <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }}/>
                       </Tooltip>
                     </span>)}
                   required={true}
                   name={`${fieldNameQuestion}[${question.key}]["title"]`}
                   validateTrigger={['onChange', 'onBlur']}
                   onChange={handleChange('title')}
                   rules={[{required: true, whitespace: true, message: "Please ask a question to the workers."}]}
        >
            <Input placeholder={this.props.placeholderQuestion} style={{width: '90%', marginRight: 8}}/>
        </Form.Item>

        {/* Questions */}
        <Form.Item {...(this.formItemLayout)}
                   label={(
                     <span>
                       Question Specific Instructions &nbsp;
                       <Tooltip title="Please specify question-specific instructions.">
                        <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
                       </Tooltip>
                     </span>)}
                   required={true}
                   name={`${fieldNameQuestion}[${question.key}]["instruction"]`}
                   validateTrigger={['onChange', 'onBlur']}
                   onChange={handleChange('instruction')}
                   rules={[{required: true, whitespace: true, message: "Please specify the instructions."}]}
        >
            <Input.TextArea placeholder="" style={{width: '90%', marginRight: 8}} autosize/>
        </Form.Item>

        
        {/* Configuration for radios. */}
        { question.type === "Radio" ?
          <DynamicItems
            form={this.props.form}
            formItemLayout={this.props.formItemLayout}
            removeByKey={this.props.removeByKey}
            addByKey={this.props.addByKey}
            updateByKey={this.props.updateByKey}
            rootKeys={this.props.rootKey.concat([question.key, "options"])}
            items={question.options}
            title="Choice"
            textHelp="Give the worker an answer option."
            textAdd="Add an choice"
            placeholder={this.props.placeholderOption}
            minimumNumber={1}
          />
          : null
        }

        {/* Configure for likert scale. */}
        { question.type === "Likert Scale" ?
          <Form.Item {...(this.formItemLayout)}
                     label={(
                       <span>
                         Likert Scale Range&nbsp;
                         <Tooltip
                           title="Maximum range of the scale">
                           <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
                         </Tooltip>
                       </span>)}
                       name={`${fieldNameQuestion}[${question.key}]["likertScale"]`}
                       validateTrigger={['onChange', 'onBlur']}
                       onChange={handleChange('likertScale')}
                       rules={[{required: true, message: "Please specify a number: 3, 5, or 7."}]}
                       >
              <InputNumber defaultValue={5} min={3} max={7} step={2}/>
          </Form.Item> : null
        }

        {this.props.noPreview ? null :
         <Form.Item
           {...this.formItemLayoutWithOutLabel}>
           <PreviewQuestion
             question={question}
             systemNames={this.props.systemNames}
           />
         </Form.Item>
        }
        
        <br/>
        <hr/>
        <br/>
        {this._showExampleConfigure(question)}
      </>
    );
  }


  _showExampleConfigure(question) {
    const fieldNameQuestion = (
      String(this.props.rootKey[0])
      + this.props.rootKey.slice(1).map((key) => (`[${key}]`)).join('')
    );
    const handleChange = (fieldName) => (event) => {
      const value = event.target !== undefined ? event.target.value : event;
      this.props.updateByKey(
        this.props.rootKey.concat([question.key]),
        {[fieldName]: value}
      )
    };
    
    if (question.type === 'Radio') {
      return (question.options || []).map((opt, i) => (<>
        <p>For choice <b><i>{opt.content}</i></b></p>
        <Form.Item {...(this.formItemLayout)}
                   label={(
                     <span>
                       Instructions specific to the choice &nbsp;
                       <Tooltip
                         title="Explain when this choice should be chosen.">
                         <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
                       </Tooltip>
                     </span>)}
                     name={`${fieldNameQuestion}[${question.key}][instructionOpt${i}]`}
                     validateTrigger={['onChange', 'onBlur']}
                     onChange={handleChange(`instructionOpt${i}`)}
                     rules={[{required: true, message: "Please specify the instructions."}]}
                     >
            <Input.TextArea placeholder="" style={{width: '90%', marginRight: 8}} autosize/>
        </Form.Item>
        
        <DynamicItems
          form={this.props.form}
          formItemLayout={this.props.formItemLayout}
          removeByKey={this.props.removeByKey}
          addByKey={this.props.addByKey}
          updateByKey={this.props.updateByKey}
          rootKeys={this.props.rootKey.concat([question.key, `exampleOpt${i}`])}
          items={question[`exampleOpt${i}`] || []}
          title={`Examples for the choice`}
          textHelp={(
            "Please provide an example for which this choice should be chosen. "
            + "It helps the workers understand when they should choose this choice, "
            + "and thus ensures better quality of your collected data."
          )}
          textHelpExplain="Please provide an explanation for why this choice should be chosen."
          textAdd="Add an example"
          placeholder={this.props.placeholderExampleAgree}
          placeholderExplain="Why should this choice be chosen?"
          recommendedNumber={3}
          recommendedMinNumber={1}
          required={true}
          minimumNumber={1}
        />

        <br/>
        <hr/>
        <br/>
      </>)
);
    } else if (question.type === 'Likert Scale') {
      return (<>
        {/* Examples. */}
        <DynamicItems
          form={this.props.form}
          formItemLayout={this.props.formItemLayout}
          removeByKey={this.props.removeByKey}
          addByKey={this.props.addByKey}
          updateByKey={this.props.updateByKey}
          rootKeys={this.props.rootKey.concat([question.key, "examples"])}
          items={question.examples || []}
          title="Example for strongly agree"
          textHelp={(
            "Please provide an example where all stars should be annotated. "
            + "It helps the worker understand what strongly agrees with the description, "
            + "and thus ensures better quality of your collected data."
          )}
          textHelpExplain="Please provide an explanation for why this example agrees with the description."
          textAdd="Add an example"
          placeholder={this.props.placeholderExampleAgree}
          placeholderExplain="Why does this example agree with the description?"
          recommendedNumber={3}
          recommendedMinNumber={1}
          required={true}
          minimumNumber={1}
        />

        <br/>
        <hr/>
        <br/>
        
        <DynamicItems
          form={this.props.form}
          formItemLayout={this.props.formItemLayout}
          removeByKey={this.props.removeByKey}
          addByKey={this.props.addByKey}
          updateByKey={this.props.updateByKey}
          rootKeys={this.props.rootKey.concat([question.key, "counterexamples"])}
          items={question.counterexamples || []}
          title="Examples for strongly disagree"
          textHelp={(
            "Please provide an example where only one star should be annotated. "
            + "It helps the worker understand what strongly disagrees with the description, "
            + "and thus ensures better quality of your collected data."
          )}
          textHelpExplain="Please provide an explanation for why this example disagrees with the description."
          textAdd="Add an example."
          placeholder={this.props.placeholderExampleDisagree}
          placeholderExplain="Why does this example disagree with the description?"
          recommendedNumber={3}
          recommendedMinNumber={1}
          required={true}
          minimumNumber={1}
        />
      </>);
    } else {
      return (<>
        {/* Examples. */}
        <DynamicItems
          form={this.props.form}
          formItemLayout={this.props.formItemLayout}
          removeByKey={this.props.removeByKey}
          addByKey={this.props.addByKey}
          updateByKey={this.props.updateByKey}
          rootKeys={this.props.rootKey.concat([question.key, "examples"])}
          items={question.examples || []}
          title="Example"
          textHelp={(
            "Please provide an example of an answer to your question above. "
            + "It helps the worker understand what is an acceptable answer, "
            + "and thus ensures better quality of your collected data."
          )}
          textHelpExplain="Please provide an explanation for why this is an acceptable answer for the question."
          textAdd="Add an example"
          placeholder={this.props.placeholderExample}
          placeholderExplain="Why is this example well done?"
          recommendedNumber={3}
          recommendedMinNumber={1}
          required={question.type === 'Open ended'}
          minimumNumber={question.type === 'Open ended' ? 1 : 0}
        />

        <br/>
        <hr/>
        <br/>
        
        <DynamicItems
          form={this.props.form}
          formItemLayout={this.props.formItemLayout}
          removeByKey={this.props.removeByKey}
          addByKey={this.props.addByKey}
          updateByKey={this.props.updateByKey}
          rootKeys={this.props.rootKey.concat([question.key, "counterexamples"])}
          items={question.counterexamples || []}
          title="Counterexample"
          textHelp={("Please provide an counterexample of an answer to your question above. "
                   + "It helps the worker understand what answers are unacceptable, "
                   + "and thus reduces low quality responses in your collected data."
          )}
          textHelpExplain="Please provide an explanation for why this is not an acceptable answer for the question"
          textAdd="Add a counterexample"
          placeholder={this.props.placeholderCounterexample}
          placeholderExplain="Why is this example badly done?"
          recommendedNumber={3}
          recommendedMinNumber={1}
          required={question.type === 'Open ended'}
          minimumNumber={question.type === 'Open ended' ? 1 : 0}
        />
      </>);
    }
  }
    
  _showQuestionType (question) {
    const fieldNameQuestion = (
      String(this.props.rootKey[0])
      + this.props.rootKey.slice(1).map((key) => (`[${key}]`)).join('')
    );
    
    if (this.constructor.questionTypes.length === 0) {
      return null;
    } else {
      return (
        <Form.Item {...this.formItemLayout} label="Type of Question"
          name={`${fieldNameQuestion}[${question.key}]["type"]`}
          rules={[{required: true, message: "Select one of them"}]}
        >
        {getFieldDecorator(
          `${fieldNameQuestion}[${question.key}]["type"]`,
          {
            initialValue: question.type,
            rules: [{
              required: true,
              message: "Select one of them"
            }]
          }
        )(
          <Radio.Group onChange={
            (e) => {
              this.props.updateByKey(
                this.props.rootKey.concat([question.key]),
                {type: e.target.value}
              )
            }}
          name={question.key.toString()}>
          {this.constructor.questionTypes.map(
            ([value, description, explanation]) => (
              <Radio value={value}>
              <span>
              {description} &nbsp;
              <Tooltip title={explanation}>
              <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
              </Tooltip>
              </span>
              </Radio>
            )
          )}
          </Radio.Group>
        )}      
        </Form.Item>
      );
    }
  }
}


class SurveyQuestionList extends QuestionList {
  static questionTypes = [
    ["Voting", "A/B tests", "Select the system that performed better on a specific task you specify."],
    ["Likert Scale", "Likert Scale",
     "Give the system a score from 1 to 5, 1 as strongly disagree and 5 as strongly agree."],
    ["Open ended", "Open ended",
     "Require more thought and more than a simple one-word answer."],
    ["Radio", "Multiple Choice",
     "Multiple choices are given."]
  ];   
}


class DynamicItems extends React.Component {
  /* Props:
   * @{object}	form:
   * @{style}	formItemLayout:
   * @{function} removeByKey: Function that can remove a element by id.
   * @{function} addByKey: Function that can remove a element by id.
   * @{function} updateByKey: Function that can update a element by id.
   * @{Array} rootKeys: Keys to update parent state.
   * @{Array} items: `[{key: @{int}, content: @{string}}]`, items to show.
   * @{string} title: 
   * @{string} textHelp: 
   * @{string} textAdd: 
   * @{string} placeholder: 
   * @{int} recommendedNumber:      
   * @{int} minimumNumber: Default 1.
   */
  constructor (props) {
    super(props);
    this.styleWarning = {
      'width': '90%', 'color': 'darkorange',
      'text-align': 'center',
      'font-weight': 'bold'
    };
  }
  
  render () {
    const {fields, rootKeys, placeholder, placeholderExplain, textHelp,
           recommendedNumber, recommendedMinNumber, textAdd, items, title} = this.props;
    const formItemLayoutWithOutLabel = {
      wrapperCol: {
        xs: {span: 24, offset: 0},
        sm: {span: 20, offset: 4},
      },
    };
    /* Show dynamic fields, and the adding button as well. */
    return (
      <>
        {/* Warning message when the number of fields is too few. */}
        {
          (recommendedMinNumber !== undefined && items.length < recommendedMinNumber) ?
          <div style={{width: "100%", textAlign: "center"}}>
            <span style={this.styleWarning}>
              <ExclamationCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
          &nbsp;
          Having at least {recommendedMinNumber} example is recommended.
            </span>
          </div>  : null
        }

      {/* The dynamic fields. */}
        { this._showDynamicInputs(items, rootKeys, placeholder, placeholderExplain) }

        {/* The add button. */}
        <Form.Item
          {...formItemLayoutWithOutLabel}>
          <Button type="dashed"
                  onClick={() => {this.props.addByKey(rootKeys)}}
                  style={{width: '90%'}}>
            <PlusOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} /> {textAdd}
          </Button>
          {/* Warning message when the number of fields is too many. */}
          {
            (recommendedNumber !== undefined && items.length > recommendedNumber) ?
            <div style={this.styleWarning}>
              <ExclamationCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
              Adding more than {recommendedNumber} examples is not recommended.
            </div> : null
          }
          
        </Form.Item>
      </>
    );
  }

  _showDynamicInputs (fields, keys, placeholder = "", placeholderExplain = undefined) {
    /** 
     * Render example(s) based on the type of `example[k]`. `example[k]` can
     * a string or an array of string.
     *
     * @param {Array}	    examples			Examples. Each element in it is {key: int, content: string}.
     * @param {Array}		keys				Path to the fields, so the parent can access the fields by
     * `self.state[keys[0]][keys[1]][...]`.
     * @param {String}	placeholder				Placeholder for the fields.
     * @param {String}	placeholderExplain		If set, a filed for explanation will be shown
     **/
    const {removeExample} = this.props;

    // Workround (?)
    if (fields === undefined){
      fields = []
    }

    // Generate the field name in the HTML form.
    let fieldName = keys[0].toString();
    for (let key of keys.slice(1)) {
      fieldName = fieldName + `[${key}]`;
    }

    // Elements to show
    const children = []
    for (const field of fields) {
      const fieldSuffix = placeholderExplain === undefined ? "" : '["content"]';
      const width = "90%";
      const handleChange = (event) => {
        this.props.updateByKey(
          keys.concat([field.key]),
          {content: event.target.value}
        )
      };
      const handleChangeExp = (event) => {
        this.props.updateByKey(
          keys.concat([field.key]),
          {explain: event.target.value}
        )
      };

      children.push(
        <Form.Item
          {...(this.props.formItemLayout)}
          label={(
            <span>
              {this.props.title} &nbsp;
              <Tooltip
                title={this.props.textHelp}>
                <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
              </Tooltip>
            </span>)}
        >
          <div>
          <div style={{display: 'inline-block', width: width, height: "100%"}}>
          <Form.Item 
            required={this.props.required !== undefined ? this.props.required : true}
            name={`${fieldName}[${field.key}]${fieldSuffix}`}
            validateTrigger={['onChange', 'onBlur']}
            rules={[{required: true, whitespace: true, message: "Please input some text."}]}
            onChange={handleChange}
          >
            <Input.TextArea
              placeholder={placeholder} style={{marginRight: 8}}
              autoSize
            />
          </Form.Item>
          </div>
          <div style={{display: 'inline-block', 'margin-left': '15px'}}>
          {fields.length > (this.props.minimumNumber === undefined ? 1 : this.props.minimumNumber) ?
                           <span>
                           <Tooltip>
                             <a onClick={
                             (
                               (id) => () => (
                                 this.props.removeByKey(keys.concat([id]))
                               )
                             )(field.key)
                             }>
                               <MinusCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle', paddingBottom: '50%'}} />
                             </a>
                           </Tooltip> </span>: null        
          }   
          </div>  </div>     
        </Form.Item>
      );
      
      // show filed for explain
      if (placeholderExplain !== undefined) {
        children.push(
          <Form.Item
            {...(this.props.formItemLayout)}
            label={(
              <span>
                Explanation &nbsp;
                <Tooltip
                  title={this.props.textHelpExplain}>
                  <QuestionCircleOutlined style={{ display: 'inline-block', verticalAlign: 'middle' }} />
                </Tooltip>
              </span>)}
          >
            <Form.Item 
              required={this.props.required !== undefined ? this.props.required : true}
              name={`${fieldName}[${field.key}]["explain"]`}
              validateTrigger={['onChange', 'onBlur']}
              rules={[{whitespace: true, message: "Please input an explanation."}]}
              onChange={handleChangeExp}
              >
                <Input.TextArea
                  placeholder={placeholderExplain} style={{width: width, marginRight: 8}}
                  autoSize
                />
              </Form.Item>
          </Form.Item>
        );
      }
    }
    return children;
  }

}

export default QuestionList;
export {lists2Questions, addKeys, SurveyQuestionList, DynamicItems};
