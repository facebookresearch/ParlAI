/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from 'react';
import {connect} from 'react-redux';
import CategoryConfigure from "./CategoryConfigure.js";

class Category extends React.Component {
  /* Props:
     {@function} dispatch
     {@Array} session
  */

  constructor(props) {
    super(props);
    this.state = {};
  }

  componentDidMount() {
    console.log("Category open")
  }

  render() {
    return <div>
      <CategoryConfigure data={this.props.session}/>
    </div>
  }
}

function mapStateToProps(state) {
  return {
    session: state.session_category,
  }
}

export default connect(mapStateToProps)(Category);
