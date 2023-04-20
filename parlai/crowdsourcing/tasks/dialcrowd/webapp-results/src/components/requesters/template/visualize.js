/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import React, { Component } from "react";
import { Tooltip } from "antd";
import { XYPlot, XAxis, YAxis, HeatmapSeries, LabelSeries } from "react-vis";
import { interpolateViridis } from "d3-scale-chromatic";

/* eslint-disable no-unused-vars */

class Bar extends Component {
  /* Visualize ratios.
   * Props:
   * {@Array} ratios: Ratios of the constitution.
   * {@Array} values: Titles corresponding to each ratio.
   * {@String} height
   * {@String} width
   * {@String} minWidth
   */
  render() {
    // const colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red'];
    const colors = [
      "#8F63F7",
      "#6089D6",
      "#6AE9EB",
      "#72D493",
      "#B6F587",
      "#F7D14F",
      "#D6934F",
      "#EB6E57",
      "#D461B1"
    ];
    let segments = [];
    let legends = [];
    for (const [i, ratio] of this.props.ratios.entries()) {
      const width = `${parseInt(ratio * 100)}%`;
      const backgroundColor = colors[i % colors.length];
      let text = "";
      if (this.props.values === undefined) {
        text = width;
      } else {
        text = `${this.props.values[i]}: ${width}`;
      }
      segments.push(
        <Tooltip placement="bottom" title={text}>
          <div style={{ ...barStyle, width, backgroundColor }}></div>
        </Tooltip>
      );

      let legendText = "";
      if (this.props.values !== undefined) {
        if (this.props.values[i].split(" ") > 3) {
          legendText =
            this.props.values[i]
              .split(" ")
              .splice(0, 3)
              .join(" ") + " ... ";
        } else {
          legendText = this.props.values[i];
        }
        legendText = legendText + `: ${width}`; // Ex. xxx: 10%
        legends.push({
          color: backgroundColor,
          text: legendText,
          ratio: ratio
        });
      }
    }
    legends = legends.sort((a, b) => (a.ratio > b.ratio ? -1 : 1)).splice(0, 3);

    return (
      <div>
        <div
          style={{
            width: this.props.width || "100%",
            height: this.props.height || "10px",
            minWidth: this.props.minWidth,
            marginBottom: "5px"
          }}
        >
          {segments}
        </div>
        <div>
          {/* eslint-disable react/jsx-key */}
          {legends.map(legend => (
            <span>
              {" "}
              <span style={{ color: legend.color }}>⬤</span> {legend.text}{" "}
              &nbsp;{" "}
            </span>
          ))}
          {/* eslint-enable react/jsx-key */}
        </div>
      </div>
    );
  }
}

class ConfusionMatrix extends Component {
  /* Props:
   * {@Array} annotations: Array of array. Each array contains annotations of a task unit.
   * {@Array} options: Possible answers.
   */
  render() {
    const [seriesMatrix, seriesLabel] = this.makeMatrix(
      this.props.annotations,
      this.props.options
    );
    return (
      <XYPlot
        colorType="literal"
        xType="ordinal"
        xDomain={this.props.options}
        yType="ordinal"
        yDomain={[...this.props.options].reverse()}
        margin={50}
        width={500}
        height={500}
      >
        <XAxis
          style={{
            text: { "font-size": "16pt" }
          }}
          orientation="top"
        />
        <YAxis
          style={{
            text: { "font-size": "16pt" }
          }}
        />
        <HeatmapSeries
          style={{
            stroke: "white",
            strokeWidth: "2px"
          }}
          data={seriesMatrix}
        />
        <LabelSeries
          style={{
            pointerEvents: "none",
            fill: "white",
            "font-size": "14pt"
          }}
          data={seriesMatrix}
          labelAnchorX="middle"
          labelAnchorY="baseline"
          getLabel={d => `${d.fixed}`}
          getColor={d => "white"}
        />
      </XYPlot>
    );
  }

  makeMatrix(annotations, options) {
    // initialize a 2D array
    let matrix = Array(options.length)
      .fill(0)
      .map(i => Array(options.length).fill(0));
    let sum = 0;
    for (const annotation of annotations) {
      for (const [i1, a1] of annotation.entries()) {
        for (const [i2, a2] of annotation.entries()) {
          if (i1 <= i2) continue;
          const ia1 = options.indexOf(a1);
          const ia2 = options.indexOf(a2);
          if (ia1 === -1 || ia2 === -1) {
            continue;
          }
          matrix[ia1][ia2] += 1;
          if (ia1 !== ia2) {
            matrix[ia2][ia1] += 1;
          }
          sum += 1;
        }
      }
    }
    let seriesMatrix = [];
    for (let i = 0; i < options.length; i += 1) {
      for (let j = 0; j < options.length; j += 1) {
        let entry = {
          x: options[i],
          y: options[j],
          color: interpolateViridis((2 * matrix[i][j]) / (sum + 1e-10) + 0.01),
          /* color: 2 * matrix[i][j] / (sum + 1e-10), */
          fixed: (matrix[i][j] / sum + 1e-10).toFixed(2)
        };
        seriesMatrix.push(entry);
      }
    }
    return [seriesMatrix, null];
  }
}

const barStyle = {
  margin: "0px",
  display: "inline-block",
  height: "100%"
};

/* eslint-enable no-unused-vars */

export { Bar, ConfusionMatrix };
