/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React from "react";
import { Input, Form, Button } from "antd";
import { SketchPicker } from "react-color";

/* eslint-disable  no-unused-vars */

class ColorPicker extends React.Component {
  /* Args:
     form
     {@String} name: name of the input field.
     {@String} initialValue:
     {@String} previewText
   */

  constructor(props) {
    super(props);
    this.state = {
      visiblePicker: false,
      color: this.props.initialValue
    };
    this.color = this.props.initialValue;
  }

  handleClick = () => {
    this.setState({ visiblePicker: !this.state.visiblePicker });
  };

  componentDidUpdate(prevProps) {
    if (prevProps.initialValue !== this.props.initialValue) {
      this.setState({ color: this.props.initialValue });
    }
  }

  close() {
    this.setState({
      visiblePicker: false
    });
    if (this.color !== undefined) {
      if (this.color.hex === undefined) {
        this.setState({
          color: this.color
        });
        this.props.updateByKey(["style", "global"], {
          backgroundColor: this.color.toUpperCase()
        });
      } else {
        this.setState({
          color: this.color.hex
        });
        this.props.updateByKey(["style", "global"], {
          backgroundColor: this.color.hex.toUpperCase()
        });
      }
    }
    console.log(this.props);
  }

  handleChange(e) {
    e.preventDefault();
    this.setState({ color: e.target.value });
  }

  render() {
    let colorStyle;
    if (this.props.previewText === undefined) {
      colorStyle = {
        background: this.state.color,
        color: this.state.color,
        "border-style": "solid",
        "border-width": "1px",
        "margin-left": "5px",
        "border-color": "#BBBBBB"
      };
    } else {
      colorStyle = {
        color: this.state.color
      };
    }

    return (
      <>
        {this.showPicker()}
        <div>
          <div style={{ display: "inline-block", width: "20%" }}>
            <Form.Item
              name={this.props.name}
              rules={[
                {
                  required: true,
                  whitespace: true,
                  message: "Please specify a color."
                }
              ]}
              onChange={e => this.handleChange(e)}
              validateTrigger={["onChange", "onBlur"]}
            >
              <Input
                placeholder="#000000"
                style={{
                  width: "1em",
                  height: "1em",
                  marginRight: "0.5em",
                  color: "transparent",
                  background: this.props.initialValue
                }}
                onClick={this.handleClick}
              />
            </Form.Item>{" "}
          </div>
        </div>
      </>
    );
  }

  handleColorChange(color, event) {
    this.color = color;
  }

  showPicker() {
    const popover = {
      position: "absolute",
      zIndex: "2"
    };

    const inner = {
      left: 0,
      bottom: 0,
      position: "absolute"
    };

    if (this.state.visiblePicker) {
      return (
        <>
          <div style={popover}>
            <div style={inner}>
              <SketchPicker
                color={this.state.color}
                onChangeComplete={color => this.handleColorChange(color)}
              />
              <Button onClick={() => this.close()} style={{ width: "100%" }}>
                Done
              </Button>
            </div>
          </div>
        </>
      );
    } else {
      return null;
    }
  }
}

/* eslint-enable  no-unused-vars */

export default ColorPicker;
