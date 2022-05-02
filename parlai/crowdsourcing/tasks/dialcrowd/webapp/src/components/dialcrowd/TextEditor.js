/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React, {Component} from 'react'
import {findDOMNode} from 'react-dom'
import {connect} from 'react-redux'
import getColor from './getColor'
import {Input} from 'antd'

const styles = {
  text: {},
  highlightText: {
    color: 'transparent',
    pointerEvents: 'none',
    padding: '5px 8px',
  },
  zeroPos: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  input: {
    background: 'none',
    border: 'none',
    width: '100%',
  }
};


class TextEditor extends Component {
  selectionAnchorNode: Node;
  inputNode: HTMLInputElement;

  componentDidMount() {
    console.log(this.props);
    document.addEventListener('selectionchange', () => {
      const selection = window.getSelection()

      if (
          selection.anchorNode
          && selection.anchorNode === this.selectionAnchorNode
      ) {
        /*
        setSelection(
            example.id,
            this.inputNode.selectionStart,
            this.inputNode.selectionEnd,
        )*/
      }
    }, false)
  }

  handleTextChange(event: Object) {
    const {
      example,
      edit,
    } = this.props
    const {
      text: oldText,
      entities: oldEntities,
    } = example
    const text = event.target.value
    const entities = []

    //update the entity boudaries

    oldEntities.forEach(oldEntity => {
      const oldSelection = oldText.substr(
          oldEntity.start,
          oldEntity.end - oldEntity.start,
      )

      function findClosestStart(lastMatch: ?number) {
        if (!lastMatch) {
          const index = text.indexOf(oldSelection)
          if (index === -1) {
            return index
          }
          else {
            return findClosestStart(index)
          }
        }
        else {
          const from = lastMatch + oldSelection.length
          const index = text.indexOf(oldSelection, from)
          if (index === -1) {
            return lastMatch
          }
          const prevDiff = Math.abs(oldEntity.start - lastMatch)
          const nextDiff = Math.abs(oldEntity.start - index)
          if (prevDiff < nextDiff) {
            return lastMatch
          }
          else {
            return findClosestStart(index)
          }
        }
      }
      const start = findClosestStart()
      if (start === -1) {
        return
      }

      entities.push({
        ...oldEntity,
        start: start,
        end: start + oldSelection.length,
      })
    })

    edit(example.id, {
      text,
      entities,
    })
  }

  renderEntityHighlight(text: string, entity: Object, key: number) {
    const { entityNames } = this.props
    const start = text.substr(0, entity.start)
    const value = text.substr(entity.start, entity.end - entity.start)
    const end = text.substr(entity.end)
    console.log(value);
    return (
        <div key={key} style={{...styles.zeroPos, ...styles.highlightText, "marginLeft":"5px" }}>
          <span>{start}</span>
          <span style={{...getColor(entityNames.indexOf(entity.entity))}}>
          {value}
        </span>
          <span>{end}</span>
        </div>
    )
  }

  render() {
    const { example, style } = this.props
    const { text, entities = [] } = example

    return (
        <div style={{ width: '100%', ...style }}>
          <div
              style={{position: 'relative'}}
              ref={node => this.selectionAnchorNode = node}
          >
            <Input.TextArea
                ref={node => this.inputNode = node && findDOMNode(node)}
                value={text}
                placeholder='text'
            />
            {entities.map((entity, index) => {
              return this.renderEntityHighlight(text, entity, index)
            })}
          </div>
        </div>
    )
  }
}


export default connect(null, null)(TextEditor)