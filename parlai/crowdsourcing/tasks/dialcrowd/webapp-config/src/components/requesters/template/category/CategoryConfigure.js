/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React from "react";
import { Button, Form, InputNumber, message, Table, Tooltip } from "antd";
import { QuestionCircleOutlined, UploadOutlined } from "@ant-design/icons";
import { connect } from "react-redux";
import "react-datasheet/lib/react-datasheet.css";
import FileReaderInput from "react-file-reader-input";
import Configure from "../Configure.js";
import QuestionList, { addKeys, lists2Questions } from "../QuestionList.js";

/* eslint-disable no-unused-vars */

const FormItem = Form.Item;

class CategoryConfigure extends Configure {
  /* Props:
     {@Array} data
     {@function} dispatch
     {@Array} session
  */

  onFinish = values => {
    this._saveAsJSON(this.state);
  };

  handleFileInputChange(_, results, targetStateProperty) {
    const [e, file] = results[0];
    let data1 = [];
    let lines = e.target.result.split("\n");
    if (targetStateProperty === "category_data") {
      for (let i = 0; i < lines.length; i++) {
        if (lines[i] === "") continue;
        let dic = {};
        dic["sentence"] = lines[i];
        dic["sentid"] = i + 1;
        dic["category"] = [];
        data1.push(dic);
      }
    } else {
      for (let i = 0; i < lines.length; i += 2) {
        if (lines[i] === "") continue;
        let dic = {};
        dic["sentence"] = lines[i];
        dic["sentid"] = i + 1;
        dic["answer"] = lines[i + 1];
        data1.push(dic);
      }
    }

    if (data1.length > 0) {
      message.success(data1.length + " sentences are loaded!");
    }
    this.setState({ [targetStateProperty]: data1 });
  }

  constructor(props) {
    super(props);
    this.state = {
      ...this.state,
      results_detail: [],
      hoveredCell: false,
      visible: false,
      visible_survey: false,
      taskid: "",
      category_data: [],
      results: [],
      visible_kappa: false,
      questionCategories: []
    };
    this.taskName = "category";
  }

  makeProps() {
    if (this.props.session.questionCategories === undefined) {
      // Workaround
      const categories = this.props.session.classLabel || [];
      let questions = lists2Questions(
        categories,
        categories.map(_ => ""),
        categories.map(_ => []),
        this.props.session.classExample,
        this.props.session.classCounterexample
      );
      this.setState({ questionCategories: questions });
    } else {
      this.setState({
        questionCategories: addKeys(this.props.session.questionCategories)
      });
    }
    this.setState({
      category_data: this.props.session.category_data,
      dataGolden: this.props.session.dataGolden
    });
    super.makeProps();
  }

  render() {
    const { loading } = this.state;
    const textStyleExtras = [
      {
        name: "Utterance",
        fieldName: "utterance",
        explain: "Set the style of utterance to be annotated."
      }
    ];

    const style = this.state.style || { global: {} };

    if (loading) {
      return <div></div>;
    } else {
      let initialVal = {};

      for (const val of [
        "generic_introduction",
        "generic_instructions",
        "enableMarkdown",
        "time",
        "payment",
        "interface",
        "nUnit",
        "nAssignment",
        "numofsent",
        "nUnitDuplicated",
        "nUnitGolden",
        "hasFeedbackQuestion"
      ]) {
        switch (val) {
          case "enableMarkdown":
            if (typeof this.state[val] == "undefined") {
              initialVal[val] = false;
            }
            break;
          case "nUnitDuplicated":
            if (typeof this.state[val] == "undefined") {
              initialVal[val] = "0";
            } else {
              initialVal[val] = this.state[val];
            }
            break;
          case "nUnitGolden":
            if (typeof this.state[val] == "undefined") {
              initialVal[val] = "0";
            } else {
              initialVal[val] = this.state[val];
            }
            break;
          default:
            initialVal[val] = this.state[val];
        }
      }

      console.log(this.state);
      console.log(initialVal);

      // initialize for consent question list
      if (
        this.state.requirements !== undefined &&
        this.state.requirements.length > 0
      ) {
        this.state.requirements.map(
          checkbox => (
            (initialVal[`checkbox[${checkbox.key}]`] = false),
            (initialVal[
              `requirements[${checkbox.key}]`
            ] = `${checkbox.content}`)
          )
        );
      }

      // initialize for intent questions list
      if (
        this.state.questionCategories !== undefined &&
        this.state.questionCategories.length > 0
      ) {
        this.state.questionCategories.map(
          question => (
            (initialVal[
              `questionCategories[${question.key}]["title"]`
            ] = `${question.title}`),
            (initialVal[
              `questionCategories[${question.key}]["instruction"]`
            ] = `${question.instruction}`)
          )
        );
        this.state.questionCategories.forEach(function(item, i) {
          item.examples.forEach(function(subitem, j) {
            initialVal[
              `questionCategories[${item.key}][examples][${subitem.key}]["content"]`
            ] = `${subitem.content}`;
            initialVal[
              `questionCategories[${item.key}][examples][${subitem.key}]["explain"]`
            ] = `${subitem.explain}`;
          });
          item.counterexamples.forEach(function(subitem, j) {
            initialVal[
              `questionCategories[${item.key}][counterexamples][${subitem.key}]["content"]`
            ] = `${subitem.content}`;
            initialVal[
              `questionCategories[${item.key}][counterexamples][${subitem.key}]["explain"]`
            ] = `${subitem.explain}`;
          });
        });
      }

      // initialize for appearance config

      if (
        this.state.style !== undefined &&
        Object.keys(this.state.style).length > 0
      ) {
        Object.keys(this.state.style).map((key, index) => {
          Object.keys(this.state.style[key]).map((key2, index2) => {
            switch (key) {
              case "global":
                switch (key2) {
                  case "backgroundColor":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "#FFFFFF"}`;
                    break;
                  default:
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "0"}`;
                }
                break;
              default:
                switch (key2) {
                  case "color":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "#000000"}`;
                    break;
                  case "fontFamily":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "Helvetica"}`;
                    break;
                  case "fontSize":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "12"}`;
                    break;
                  case "lineHeight":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "1.5"}`;
                    break;
                  case "text-align":
                    initialVal[`['style']['${key}']['${key2}']`] = `${this.state
                      .style[key][key2] || "left"}`;
                    break;
                }
            }
          });
        });
      }

      return (
        <div>
          <h2 style={{ "padding-left": "1%" }}>
            Template for an Intent Classification Task{" "}
          </h2>
          <p style={{ "padding-left": "1%" }}>
            This template is used for the creation of tasks that require the
            workers to label the intent of utterances.
          </p>
          <Form
            id="configform"
            initialValues={initialVal}
            onFinish={this.onFinish}
          >
            {this._showGeneralConfig()}
            {this._showAnnotationConfig("utterance")}
            {this._showDataConfig()}
            {this._showQualityControlConfig("utterance")}
            {this._showConsentConfig()}
            {this._showCategoryConfig()}
            {this._showFeedbackConfig()}
            {this._showAppearanceConfig(textStyleExtras)}
            {this._showButtons()}
          </Form>
        </div>
      );
    }
  }

  _showDataConfig() {
    const { formItemLayout, formItemLayoutWithOutLabel } = this;
    return (
      <>
        <FormItem
          {...formItemLayout}
          label={
            <span>
              Num of sentences per page&nbsp;
              <Tooltip title="Sentences per worker per task">
                <QuestionCircleOutlined
                  style={{ display: "inline-block", verticalAlign: "middle" }}
                />
              </Tooltip>
            </span>
          }
          name="numofsent"
        >
          <InputNumber
            min={1}
            max={100}
            style={{ height: "100%" }}
            onChange={e => {
              this.setState({ numofsent: e });
            }}
          />
        </FormItem>
        {this._showDataUpload()}
      </>
    );
  }

  _showDataUpload(golden = false) {
    const { formItemLayout, formItemLayoutWithOutLabel } = this;
    let columns_dialog = [
      {
        title: "ID",
        dataIndex: "sentid",
        key: "sendid",
        width: 100
      },
      {
        title: "Text",
        dataIndex: "sentence",
        key: "sentence"
      }
    ];
    if (golden) {
      columns_dialog.push({
        title: "Answer",
        dataIndex: "answer",
        key: "answer"
      });
    }

    const explain = golden ? (
      <>
        <div>Please format your data as below, separated with new lines:</div>
        <div>Utterance 1. (Ex. I want to buy a ticket.)</div>
        <div>Intent of utterance 1. (Ex. Purchase)</div>
        <div>Utterance 2. (Ex. I want to buy a ticket.)</div>
        <div>Intent of utterance 2. (Ex. Purchase)</div>
        <div>...</div>
      </>
    ) : (
      "Please split the utterances by new line."
    );
    const targetStateProperty = golden ? "dataGolden" : "category_data";

    return (
      <>
        <FormItem
          {...formItemLayout}
          label={
            <span>
              Upload your data&nbsp;
              <Tooltip title={explain}>
                <QuestionCircleOutlined
                  style={{ display: "inline-block", verticalAlign: "middle" }}
                />
              </Tooltip>
            </span>
          }
        >
          <FileReaderInput
            as="text"
            onChange={(e, results) =>
              this.handleFileInputChange(e, results, targetStateProperty)
            }
          >
            <Button style={{ width: "90%" }}>
              <UploadOutlined
                style={{ display: "inline-block", verticalAlign: "middle" }}
              />{" "}
              Click to Upload
            </Button>
          </FileReaderInput>
        </FormItem>

        {(this.state[targetStateProperty] || []).length > 0 ? (
          <div height={500}>
            <Table
              rowKey="sentence"
              dataSource={this.state[targetStateProperty]}
              columns={columns_dialog}
              pagination={{ hideOnSinglePage: true }}
              size="small"
              bordered
            />
          </div>
        ) : null}
      </>
    );
  }

  _showCategoryConfig() {
    const { formItemLayout, formItemLayoutWithOutLabel } = this;
    const instruction =
      "In this section, you can set up the types of intents the worker can choose from. " +
      "Remember to include examples and counterexamples. They help the worker get a " +
      "better idea what should be labeled as what type of intent, so you can collect data " +
      "of better quality.";
    return (
      <>
        <h3>Intent Type Configuration</h3>
        <p>{instruction}</p>
        <CategoryQuestionList
          form={this.props.form}
          formItemLayout={formItemLayout}
          removeByKey={this.removeByKey}
          addByKey={this.addByKey}
          updateByKey={this.updateByKey}
          questions={this.state.questionCategories}
          rootKey={["questionCategories"]}
          questionFieldLabel="Intent type"
          questionHelpText={"Add an intent type that the worker can select."}
          textAddQuestion="Add an Intent Type"
          textInstruction={instruction}
          placeholderQuestion="order_food"
          placeholderExample="I would like to order a pizza from Domino's."
          placeholderCounterexample="What's the temperature now?"
          noPreview={true}
        />
      </>
    );
  }

  _showButtons() {
    const { formItemLayout, formItemLayoutWithOutLabel } = this;
    return (
      <Button
        form="configform"
        key="submit"
        htmlType="submit"
        style={{ width: "90%" }}
      >
        Save Configuration as JSON
      </Button>
    );
  }
}

class CategoryQuestionList extends QuestionList {
  static questionTypes = [];
}

function mapStateToProps(state) {
  return {
    session: state.session_category
  };
}

/* eslint-enable no-unused-vars */

export default connect(mapStateToProps)(CategoryConfigure);
