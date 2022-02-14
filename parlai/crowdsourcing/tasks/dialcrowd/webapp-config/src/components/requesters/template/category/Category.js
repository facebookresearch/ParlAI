/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from 'react';
import {connect} from 'react-redux';
import {Tabs} from 'antd';
import Icon from '@ant-design/icons';
import CategoryConfigure from "./CategoryConfigure.js";

const TabPane = Tabs.TabPane;

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
      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><Icon type="home"/>Configure</span>} key="1">
          <CategoryConfigure data={this.props.session}/>
        </TabPane>
      </Tabs>
    </div>
  }
}

function mapStateToProps(state) {
  return {
    session: state.session_category,
  }
}

export default connect(mapStateToProps)(Category);
