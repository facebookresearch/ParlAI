/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from 'react';
import {Input, Form, InputNumber, Button, Select, Radio, Tooltip} from 'antd';
import {AlignLeftOutlined, AlignCenterOutlined, AlignRightOutlined,
        FontColorsOutlined, FontSizeOutlined, LineHeightOutlined} from '@ant-design/icons';
import { SketchPicker } from 'react-color';

const FormItem = Form.Item;

class FontPicker extends React.Component {
  /* Args:
     form
     {@Array} keys: Path to the config. e.g ['style', 'instruction'].
     {@Function} updateByKey: 
     {@String} fontFamily
     {@String} fontSize
     {@String} color
     {@String} previewText
   */
  constructor (props) {
    super(props);
    this.state = {
      visiblePicker: false,
    };
    this.color = this.props.color;
  }
      
  handleClick = () => {
    this.setState(
      { visiblePicker: !this.state.visiblePicker }
    );
  };

  close () {
    // close color picker
    this.setState({
      visiblePicker: false
    });

    // update parent state
    if (this.color !== undefined) {
      if (this.color.hex === undefined){
        this.props.updateByKey(this.props.keys, {color: this.color.toUpperCase()});
      }
      else{
        this.props.updateByKey(this.props.keys, {color: this.color.hex.toUpperCase()});
      }
    }
  };

  handleColorInputChange (e) {
    /* Update parent state when the input box is changed */
    e.preventDefault();
    
  }
  
  render() {
    const stylePreviewText = {
      color: this.props.color,
      "fontFamily": this.props.fontFamily,
      "fontSize": `${this.props.fontSize}pt`,
      "padding-left": "0.5em",
      "vertical-align": "middle",
      display: 'inline-block',
    };
    const fontList = [
      'Helvetica',
      'Arial',
      'Times New Roman',
      'Impact',
      'Times',
      'Arial Black'
    ];
    const prefix = path2Name(this.props.keys);

    return (<>
      {this.showPicker()}
      <div style={{display: 'inline-block', marginTop: '-4px'}}>
        {/* color */}
        <div style={{display: 'inline-block'}}>
          <div style={{display: 'inline-block'}}>
            <Tooltip title="Text color.">
              <FontColorsOutlined
                style={{marginLeft: "0.5em", marginRight: "0.5em", color: this.props.color }}/>
            </Tooltip>
          </div>
          <div style={{display: 'inline-block'}}>
            <FormItem
              name={`${prefix}['color']`}
              validateTrigger={['onChange', 'onBlur']}
              rules={[{required: true, whitespace: true, message: "Please specify a color."}]}
              onChange={(value) => this.handleColorInputChange(value)}
              >
                <Input placeholder="#000000"
                    style={{width: "1em", height: "1em", marginRight: "0.5em",
                            color: 'transparent', background: this.props.color}}
                    onClick={this.handleClick}
                />
            </FormItem>
          </div>
        </div>

        {/* font-size */}
        <div style={{display: 'inline-block'}}>
          <div style={{display: 'inline-block'}}>
            <Tooltip title="Font size.">
              <FontSizeOutlined style={{marginLeft: "1em", marginRight: "0.5em"}}/>
            </Tooltip>
          </div>
          <div style={{display: 'inline-block'}}>
            <FormItem
              name={`${prefix}['fontSize']`}
              validateTrigger={['onChange', 'onBlur']}
              rules={[{required: true, whitespace: true, pattern: new RegExp(/^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)$/), message: "Please specify the size."}]}
            >
              <InputNumber style={{width: "4em"}} onChange={(value) => this.props.updateByKey(
                this.props.keys, {"fontSize": value}
              )}/>
            </FormItem>
          </div>
        </div>

        {/* line height */}
        <div style={{display: 'inline-block'}}>
          <div style={{display: 'inline-block'}}>
            <Tooltip title="Line height.">
              <LineHeightOutlined style={{marginLeft: "1em", marginRight: "0.5em"}}/>
            </Tooltip>
          </div>
          <div style={{display: 'inline-block'}}>
            <FormItem
              name={`${prefix}['lineHeight']`}
              validateTrigger={['onChange', 'onBlur']}
              rules={[{required: true, whitespace: true, pattern: new RegExp(/^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)$/), message: "Please specify the line height."}]}
            >
              <InputNumber step={0.25} style={{width: "4em"}} onChange={(value) => this.props.updateByKey(
                this.props.keys, {"lineHeight": value}
              )}/>
            </FormItem>
          </div>
        </div>

        {/* font-family */}
        <span style={{marginRight: "0.5em"}}></span>
        <div style={{display: 'inline-block'}}>
          <FormItem
            name={`${prefix}['fontFamily']`}
            validateTrigger={['onChange', 'onBlur']}
            rules={[{required: true, whitespace: true, message: "Please specify the font."}]}
          >
            <Select defaultValue={this.state.family}
                    style={{"fontFamily": this.props.fontFamily}} onChange={(value) => this.props.updateByKey(
                      this.props.keys, {"fontFamily": value}
                    )}>
              {fontList.map(
                (font, i) => (
                  <Select.Option
                    value={font} key={i}
                    style={{"fontFamily": font}}>
                    {font}
                  </Select.Option>
                )
              )}
            </Select>
          </FormItem>
        </div>
        
        {/* align */}
        <span style={{marginRight: "0.5em"}}></span>
        <div style={{display: 'inline-block'}}>
          <FormItem
            name={`${prefix}['text-align']`}
          >
            <Radio.Group onChange={(e) => this.props.updateByKey(
                this.props.keys, {"text-align": e.target.value}
              )}>
              <Radio.Button value="left" style={{paddingLeft: '0.5em', paddingRight: '0.5em'}}>
                <AlignLeftOutlined />
              </Radio.Button>
              <Radio.Button value="center" style={{paddingLeft: '0.5em', paddingRight: '0.5em'}}>
                <AlignCenterOutlined />
              </Radio.Button>
              <Radio.Button value="right" style={{paddingLeft: '0.5em', paddingRight: '0.5em'}}>
                <AlignRightOutlined />
              </Radio.Button>
            </Radio.Group>
          </FormItem>
        </div>
        
        <div style={{display: 'inline-block'}}>
          <span style={stylePreviewText}>{this.props.previewText || "preview"}</span>
        </div>
      </div>
    </>);
  }

  handleColorPickerChange (color, event) {
    this.color = color;
  }
  
  showPicker () {
    const popover = {
      position: 'absolute',
      zIndex: '2',
    };

    const inner = {
      left: 0,
      bottom: 0,
      position: 'absolute'
    };
    
    const cover = {
      position: 'fixed',
      top: '0px',
      right: '0px',
      bottom: '0px',
      left: '0px',
    };

    if (this.state.visiblePicker) {
      return (<>
        <div style={ popover }>
          <div style={ inner }>
            <SketchPicker
              color={this.state.color}
              onChangeComplete={(color) => this.handleColorPickerChange(color)}
            />
            <Button onClick={() => this.close()}
                    style={{width: "100%"}}>Done
            </Button>
          </div>
        </div>
      </>);
    } else {
      return null;
    }
  }
}


function path2Name (path) {
  /* Convert a path to name of a form item. 
     Args:
     {@Array} path
     Return: {@String}
   */
  let name = "";
  for (const v of path) {
    name = name + `['${v}']`;
  }
  return name;
}

export default FontPicker;

