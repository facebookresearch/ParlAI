/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React, {Component} from 'react'
import {Table} from 'antd';

let getMean = function (data) {
  return data.reduce(function (a, b) {
    return Number(a) + Number(b);
  }) / data.length;
};

let calculateFleiss = function (data, row_length, column_length) {
  var fleiss_row = new Array(column_length).fill(0);
  var fleiss_row_total = new Array(column_length).fill(0);
  var total = 0
  var row_totals = []

  for (let i = 0; i < row_length; i++){
    var partial_sum = 0
    for (let j = 0; j < column_length; j++){
      fleiss_row[j] += data[i][j]
      total += data[i][j]
      partial_sum += data[i][j]
    }
    row_totals.push(partial_sum)
  }

  var pe = 0
  for (let i = 0; i < column_length; i++){
    var value = fleiss_row[i]/total
    fleiss_row_total[i] = value
    pe += (value*value)
  }

  var pi = []
  for (let i = 0; i < row_length; i++){
    var sum = 0
    for (let j = 0; j < column_length; j++){
      sum += (data[i][j] * (data[i][j] - 1))
    }
    pi.push(sum/(row_totals[i]*(row_totals[i]-1)))
  }
  var pmean = getMean(pi)

  var total_fleiss = (pmean - pe)/(1 - pe)

  console.log(pmean)

  if (isNaN(pmean)){
    total_fleiss = 1
  }

  return total_fleiss
}

class DataStatistics extends Component {
  /* Props:
   * {@String} url: Url from which statistics can be fetched.
   * {@String} urlResults: Url from which results can be fetched.
   * {@Array} questions: Array of questions. `propertyNameQuestions` 
   * is used only when this prop is not given.
   * {@String} propertyNameQuestions: Name of the question list, e.g. questionSurveys.
   */
  static columns = [
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question'
    },
    {
      title: "Fleiss' Kappa",
      dataIndex: "Fleiss' Kappa",
      key: "Fleiss",
      render: renderWithColor
    },
  ];
  constructor (props) {
    super(props);
    this.state = {
      results: [],
      fleiss_matrix: [],
      value_keys: []
    };
  }

  componentDidMount () {
    this.getResults();
  }

  getResults() {
    fetch('http://localhost:5000/config')
    .then(
      res => res.json()
    ).then(res => {
      var value_keys = {}
      res['questionCategories'].forEach((value, i) => {
        value_keys[value['title']] = value['key']
      })
      this.setState({
        results: this.props.data,
        value_keys: value_keys,
        total_sentences: res['category_data'].length
      })
    })
  }

  makeDataSource() {
    var fleiss = []

    var column_length = Object.keys(this.state.value_keys).length

    for (let i = 0; i < this.state.total_sentences; i++){
      fleiss[i] = new Array(column_length).fill(0)
    }

    var sentences = []

    this.state.results.forEach((data, i) => {
      data.detail.forEach((sentence, j) => {
        var id = sentence['id'] - 1
        var item = this.state.value_keys[sentence['label']]
        fleiss[id][item] += 1
        sentences.push(sentence['sentence'])
      })
    })

    // proportion of the number of ratings assigned to each category
    var all_fleiss = []
    var total_fleiss = calculateFleiss(fleiss, this.state.total_sentences, column_length)
    all_fleiss.push({'sentence': 'Total Fleiss', 'fleiss': total_fleiss})
    for (let i = 0; i < this.state.total_sentences; i++){
      if (sentences[i]){
        all_fleiss.push({'sentence': sentences[i], 'fleiss': calculateFleiss(fleiss[i], 1, column_length)})
      }
    }
    
    return all_fleiss

  }

  render () {
    if (!(this.state.results).length){
      return null;
    }
    else{
      const columns = [
        {
          title: 'Sentence',
          dataIndex: 'sentence',
          key: 'sentence'
        },
        {
          title: 'fleiss',
          dataIndex: 'fleiss',
          key: 'fleiss',
          render: renderWithColor
        },
      ];

      const dataSource = this.makeDataSource();

      return (<>
        <Table dataSource={dataSource} columns={columns}
              pagination={{hideOnSinglePage: true}}
        />
        <br/>
        {this.showFleissColorCode()}
      </>);
    }
  }

  showFleissColorCode () {
    return (<>
      <div> Fleiss' Kappa indicates worker agreement of different levels:
        <ul>
          <li> <span style={{color: "#595959", "font-weight": "bold"}}> {"< 0.00"} </span> : Poor </li>
          <li> <span style={{color: "#940004", "font-weight": "bold"}}> 0.00 - 0.20 </span> : Slight </li>
          <li> <span style={{color: "#d15700", "font-weight": "bold"}}> 0.20 - 0.40 </span> : Fair </li>
          <li> <span style={{color: "#e6b314", "font-weight": "bold"}}> 0.40 - 0.60 </span> : Moderate </li>
          <li> <span style={{color: "#90ad00", "font-weight": "bold"}}> 0.60 - 0.80 </span> : Substantial </li>
          <li> <span style={{color: "#00b31a", "font-weight": "bold"}}> 0.80 - 1.00 </span> : Almost Perfect </li>
        </ul>
      </div>
    </>);
  }
}


function renderWithColor (score) {
  let color;
  score = parseFloat(score);
  if (score < 0 || isNaN(score)) { color = '#595959'; }
  else if (score < 0.2 ) { color = '#940004'; }
  else if (score < 0.4 ) { color = '#d15700'; }
  else if (score < 0.6 ) { color = '#e6b314'; }
  else if (score < 0.8 ) { color = '#90ad00'; }
  else { color = '#00b31a'; }
  return {
    props: {},
    children: <span style={{color: color, "font-weight": "bold"}}> {score} </span>
  };
}


export default DataStatistics;
export {renderWithColor};
