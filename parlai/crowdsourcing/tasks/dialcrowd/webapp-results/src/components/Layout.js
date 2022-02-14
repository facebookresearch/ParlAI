/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import React from 'react'
import {Layout as AntLayout,} from 'antd'
import Icon from '@ant-design/icons';
import Category from './requesters/template/category/CategoryQuality.js'

class Layout extends React.Component {

  render() {
    return <AntLayout style={{height: '100%', minHeight: '100vh'}}>
      <AntLayout>
        <AntLayout.Header style={{background: '#f0f2f5', padding: 0}}>
          <Icon
              style={{marginLeft: "10px", fontSize: 20}}
          />
          <div key="logo" style={{float: "right", "marginRight": "20px"}}>
            <img width={"100px"} src={"http://dialrc.org/img/site-logo.png"} alt="logo"/>
          </div>
        </AntLayout.Header>
        <AntLayout.Content style={{
          flex: 1,
          flexDirection: 'column',
          overflow: 'scroll',
          background: '#fff',
          padding: 24
        }}>
        <Category/>
        </AntLayout.Content>

        <AntLayout.Footer style={{}}>
          <strong>Copyright © 2021, DialRC, Carnegie Mellon University</strong>
          <div style={{float: "right", "marginRight": "10px"}}><a href={"mailto:kyusongl@cs.cmu.edu"}><Icon
              style={{fontSize: 20}} type="contacts"/></a> <a href={"https://github.com/dialrc/dialcrowd"}><Icon
              style={{fontSize: 20}} type="github"/></a></div>
        </AntLayout.Footer>
      </AntLayout>
    </AntLayout>
  }
}

export default Layout