/*********************************************
 * @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee
 * Carnegie Mellon University 2022
 *********************************************/

import {
  Hint,
  HorizontalGridLines,
  LineMarkSeries,
  VerticalGridLines,
  XAxis,
  XYPlot,
  YAxis,
  ChartLabel
} from "react-vis";
import "react-vis/dist/style.css";
import React, { Component } from "react";
import { Table, Tooltip } from "antd";
import { connect } from "react-redux";

import DataStatistics from "../DataStatistics";

/* eslint-disable no-unused-vars */
/* eslint-disable react/no-unescaped-entities */

let DicUser = {};

let getMean = function(data) {
  return (
    data.reduce(function(a, b) {
      return Number(a) + Number(b);
    }) / data.length
  );
};
let getSD = function(data) {
  let m = getMean(data);
  return Math.sqrt(
    data.reduce(function(sq, n) {
      return sq + Math.pow(n - m, 2);
    }, 0) /
      (data.length - 1)
  );
};

function getWorkerQuality(t) {
  /* Retrieve data from worker files */
  console.log("retrieving data");
  var label_keys = [];
  var total_sentences = 0;
  var goldendatakey = {};

  fetch("http://localhost:5000/config")
    .then(res => res.json())
    .then(res => {
      res["questionCategories"].forEach((data, i) => {
        label_keys[data["key"]] = data["title"];
      });
      total_sentences = res["category_data"].length;

      res["dataGolden"].forEach((data, i) => {
        goldendatakey[data["sentid"]] = data["answer"];
      });
    });

  fetch("http://localhost:5000/data")
    .then(res => res.json())
    .then(res => {
      var finaldata = [];

      // parse data
      var parseddata = {};
      var timings = [];
      var sentence_annotations = {};
      var input_sentences = {};

      var goldendata = {};
      var duplicatedata = {};

      res["data"].forEach((user, i) => {
        user["mephisto_data"] = JSON.parse(user["mephisto_data"]);
        var userdata = {};
        var userid = user["userId"];

        var has_times_dict = user["mephisto_data"]["times"] !== undefined;
        var run_time = -1;
        if (has_times_dict) {
          run_time =
            user["mephisto_data"]["times"]["task_end"] -
            user["mephisto_data"]["times"]["task_start"];
        } else {
          console.log(
            "Run time could not be computed, as it must be pulled from metadata."
          );
          console.log(
            "Please update the server if you want this functionality."
          );
        }

        timings.push(run_time.toFixed(2));

        userdata["time"] = run_time.toFixed(2);

        user["mephisto_data"]["inputs"].forEach((data, j) => {
          input_sentences[data["id"]] = data["sentences"][0];
          userdata[data["id"]] = {};
        });

        var passes_duplicate = { total: 0, passes: 0 };
        var passes_golden = { total: 0, passes: 0 };

        user["mephisto_data"]["outputs"]["annotations"].forEach((data, j) => {
          try {
            if (data["id"]) {
              var sentence_no = parseInt(
                data["id"].substring(
                  data["id"].indexOf("[") + 1,
                  data["id"].indexOf("]")
                )
              );
              if (sentence_no < total_sentences) {
                if (data["id"].includes('"answer"')) {
                  userdata[sentence_no]["label"] = data["value"];
                  var label = data["value"];
                  if (!(sentence_no in sentence_annotations)) {
                    sentence_annotations[sentence_no] = {};
                    sentence_annotations[sentence_no][label] = 1;
                  } else {
                    if (!(label in sentence_annotations[sentence_no])) {
                      sentence_annotations[sentence_no][label] = 1;
                    } else {
                      sentence_annotations[sentence_no][label] += 1;
                    }
                  }
                }
                if (data["id"].includes('"duration"')) {
                  userdata[sentence_no]["duration"] = data["value"];
                }
                if (data["id"].includes('"timestamp')) {
                  userdata[sentence_no]["timestamp"] = data["value"];
                }
              } else {
                // check duplicate data
                if (sentence_no < total_sentences * 2) {
                  var corresponding_id = sentence_no - total_sentences;
                  if (data["id"].includes('"answer"')) {
                    if (userdata[corresponding_id]["label"] === data["value"]) {
                      passes_duplicate["passes"] += 1;
                    }
                    passes_duplicate["total"] += 1;
                  }
                }

                // check golden data
                if (sentence_no > total_sentences * 2) {
                  /* eslint-disable no-redeclare */
                  var corresponding_id = sentence_no - total_sentences * 2;
                  /* eslint-enable no-redeclare */
                  if (data["id"].includes('"answer"')) {
                    if (goldendatakey[corresponding_id] === data["value"]) {
                      passes_golden["passes"] += 1;
                    }
                    passes_golden["total"] += 1;
                  }
                }
              }
            }
          } catch (err) {
            console.log(err);
          }
        });

        duplicatedata[userid] = passes_duplicate;
        goldendata[userid] = passes_golden;

        parseddata[userid] = userdata;
      });

      // upper and lower bound time for outliers
      var average_time = getMean(timings);
      var upperbound = average_time + getSD(timings);
      var lowerbound = average_time - getSD(timings);

      // majority
      var majority_sentence_annotation = {};
      for (const [key, value] of Object.entries(sentence_annotations)) {
        var max_val = Math.max(...Object.values(value));
        var max_keys = [];
        for (const [key2, value2] of Object.entries(value)) {
          if (value2 == max_val) {
            max_keys.push(key2);
          }
        }
        majority_sentence_annotation[key] = max_keys;
      }

      for (const [key, value] of Object.entries(parseddata)) {
        var intermediate_data = {};

        intermediate_data["userId"] = key;
        var time = Number.parseFloat(value["time"]).toFixed(2);
        intermediate_data["total_duration"] = time;

        //outlier: time too far
        if (time < lowerbound || time > upperbound) {
          intermediate_data["outlier"] = "true";
        } else {
          intermediate_data["outlier"] = "false";
        }

        //agreeself: duplicate
        intermediate_data["agreeSelf"] =
          duplicatedata[key]["passes"].toString() +
          "/" +
          duplicatedata[key]["total"].toString();

        //agreegold: golden data
        intermediate_data["agreeGold"] =
          goldendata[key]["passes"].toString() +
          "/" +
          goldendata[key]["total"].toString();

        intermediate_data["detail"] = [];

        var agree = 0;
        var total = 0;

        var durations = [];

        var label_ordered = [];

        for (const [key1, value1] of Object.entries(value)) {
          if (key1 !== "time") {
            intermediate_data["detail"].push({
              id: key1,
              sentence: input_sentences[key1],
              label: value1["label"],
              duration: value1["duration"] / 1000,
              timestamp: value1["timestamp"]
            });
            durations.push(value1["duration"] / 1000);
            try {
              if (
                majority_sentence_annotation[key1].includes(value1["label"])
              ) {
                agree += 1;
              }
              total += 1;
            } catch (err) {
              console.log(err);
            }

            label_ordered.push(value1["label"]);
          }
        }
        //agreemajor: agree with all
        intermediate_data["agreeMajor"] =
          agree.toString() + "/" + total.toString();

        //abnormal: repeating abab, all a
        var abnormal = true;

        // all a
        const allSame = val => val === label_ordered[0];
        abnormal = label_ordered.every(allSame);

        const allSameb = val => val === label_ordered[1];

        // repeating abab
        if (!abnormal) {
          const evens = label_ordered.filter((data, i) => i % 2 === 0);
          const odds = label_ordered.filter((data, i) => i % 2 === 1);
          if (evens.length > 1 || odds.length > 1) {
            abnormal = evens.every(allSame) && odds.every(allSameb);
          }
        }

        if (abnormal) {
          intermediate_data["abnormal"] = "true";
        } else {
          intermediate_data["abnormal"] = "false";
        }

        intermediate_data["average_duration"] = getMean(durations).toFixed(2);
        intermediate_data["std_duration"] = getSD(durations).toFixed(2);

        finaldata.push(intermediate_data);
      }

      t.setState({ results_detail: finaldata });
    });
}

// gaussian graph
function gaussian(arg) {
  let items = [];
  arg.map((item, i) => {
    let avg = item["average_duration"];
    let total = item["total_duration"];
    let userId = item["userId"];
    DicUser[total] = userId;
    items.push(total);
  });

  let getMean = function(data) {
    return (
      data.reduce(function(a, b) {
        return Number(a) + Number(b);
      }) / data.length
    );
  };
  let getSD = function(data) {
    let m = getMean(data);
    return Math.sqrt(
      data.reduce(function(sq, n) {
        return sq + Math.pow(n - m, 2);
      }, 0) /
        (data.length - 1)
    );
  };

  if (items.length > 0) {
    let m = getMean(items);
    let std = getSD(items);
    let a = 1 / Math.sqrt(2 * Math.PI);
    let f = a / std;
    let p = -1 / 2;
    let finallist = [];
    if (isNaN(std)) {
      std = 1;
    }
    items.sort(function(a, b) {
      return a - b;
    });
    items.map((z, i) => {
      let c = (z - m) / std;
      c *= c;
      p *= c;
      let result =
        (1 / (Math.sqrt(2 * Math.PI) * std)) *
        Math.E ** (-0.5 * ((z - m) / std) ** 2);
      finallist.push({ x: z, y: result });
    });
    return finallist;
  }
  return [{ x: 1, y: 11 }];
}

function buildValue(hoveredCell) {
  const { v, userId } = hoveredCell;
  return {
    x: v.x,
    y: v.y,
    userId: userId
  };
}

class CategoryQuality extends Component {
  constructor(props) {
    super(props);
    this.state = {
      results_detail: [],
      survey: [],
      results: [],
      grid: [],
      cost: 0,
      visible_kappa: false,
      feedback: []
    };
  }

  componentDidMount() {
    getWorkerQuality(this);
  }

  render() {
    const detail_subcolums = [
      {
        title: "Sentence",
        dataIndex: "sentence",
        key: "sentence"
      },
      {
        title: "label",
        dataIndex: "label",
        key: "label"
      },
      {
        title: "duration",
        dataIndex: "duration",
        key: "duration"
      },
      {
        title: "timestamp",
        dataIndex: "timestamp",
        key: "timestamp"
      }
    ];
    const detail_colums = [
      {
        title: "user id",
        dataIndex: "userId",
        key: "userId"
      },
      {
        title: "total",
        dataIndex: "total_duration",
        key: "total_duration"
      },
      {
        title: "average",
        dataIndex: "average_duration",
        key: "average_duration"
      },
      {
        title: "std",
        dataIndex: "std_duration",
        key: "std_duration"
      },
      {
        title: (
          <>
            outlier
            <Tooltip title="Whether the time spent on the task is 2 standard deviations above or below the average time taken.">
              <a>
                <sub>?</sub>
              </a>
            </Tooltip>
          </>
        ),
        dataIndex: "outlier",
        key: "outlier"
      },
      {
        title: (
          <>
            abnormal
            <Tooltip title="Whether some repeating pattern, like answering A, B, A, B ..., or A, A, A is detected.">
              <a>
                <sub>?</sub>
              </a>
            </Tooltip>
          </>
        ),
        dataIndex: "abnormal",
        key: "abnormal"
      },
      {
        title: (
          <>
            intra-user
            <Tooltip
              title={
                "The number of times the worker responded consistently" +
                "/ the number of duplicated task units given in each HIT."
              }
            >
              <a>
                <sub>?</sub>
              </a>
            </Tooltip>
          </>
        ),
        dataIndex: "agreeSelf",
        key: "agreeSelf"
      },
      {
        title: (
          <>
            agree-gold
            <Tooltip
              title={
                "The number of correctly answered golden task units " +
                "/ the number of golden task units given in each HIT."
              }
            >
              <a>
                <sub>?</sub>
              </a>
            </Tooltip>
          </>
        ),
        dataIndex: "agreeGold",
        key: "agreeGold"
      },
      {
        title: (
          <>
            inter-user
            <Tooltip
              title={"The ratio of annotations that agree with the majority."}
            >
              <a>
                <sub>?</sub>
              </a>
            </Tooltip>
          </>
        ),
        dataIndex: "agreeMajor",
        key: "agreeMajor"
      }
    ];

    if (!this.state.results_detail.length) {
      return null;
    } else {
      console.log(this.state.results_detail);
      return (
        <div>
          <p>
            We calculate the worker's time according to how long they take in
            selecting the intent for each question. Abnormality is described as
            any worker who selects the same intent for every sentence and
            outlier is calculated by any worker who takes more than 2 standard
            deviations longer or shorter than the mean time taken for the task.
            These two metrics can be used to help determine if a worker may be a
            bot; however, we suggest that you pay the bot, block them from
            completing future HITs for you, and{" "}
            <a href="https://support.aws.amazon.com/#/contacts/aws-mechanical-turk">
              report
            </a>{" "}
            it to Amazon, which can also be done through "Report this HIT" on
            the MTurk interface. This is because rejecting the bot may hurt your
            requester reputation. Be sure to accept and reject your HITs in a
            timely manner and communicate with the workers if they have any
            questions. You can monitor your requester reputation on sites such
            as <a href="https://turkopticon.ucsd.edu/">Turkopticon</a>.
          </p>
          <h1>Workers' Timestamps Logs</h1>
          <Table
            rowKey="userId"
            dataSource={this.state.results_detail}
            columns={detail_colums}
            size="small"
            expandedRowRender={record => (
              <Table dataSource={record.detail} columns={detail_subcolums} />
            )}
          />

          <h1>Agreement between Workers</h1>
          <DataStatistics data={this.state.results_detail} />

          <h1>Gaussian of Timestamps</h1>
          <XYPlot width={300} height={300}>
            <VerticalGridLines />
            <HorizontalGridLines />
            <XAxis />
            <YAxis />
            <ChartLabel
              text="total time to select intents"
              className="alt-x-label"
              includeMargin={false}
              xPercent={0.25}
              yPercent={1.01}
            />
            <ChartLabel
              text="probability density"
              className="alt-y-label"
              includeMargin={false}
              xPercent={0.06}
              yPercent={0.06}
              style={{
                transform: "rotate(-90)",
                textAnchor: "end"
              }}
            />
            <LineMarkSeries
              className="linemark-series-example-2"
              curve={"curveMonotoneX"}
              data={gaussian(this.state.results_detail)}
              onValueMouseOver={v =>
                this.setState({
                  hoveredCell:
                    v.x && v.y ? { v: v, userId: DicUser[v.x] } : false
                })
              }
            />
            {this.state.hoveredCell ? (
              <Hint value={buildValue(this.state.hoveredCell)}>
                <div style={{ color: "black" }}>
                  <b>
                    <strong>
                      {" "}
                      {"userId: " + this.state.hoveredCell.userId}
                    </strong>
                  </b>
                </div>
              </Hint>
            ) : null}
          </XYPlot>
        </div>
      );
    }
  }
}

function mapStateToProps(state) {
  return {
    session: state.session_category
  };
}

/* eslint-enable no-unused-vars */
/* eslint-enable react/no-unescaped-entities */

export default connect(mapStateToProps)(CategoryQuality);
export { getWorkerQuality };
