/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import {Provider} from 'react-redux';
import Layout from "./components/Layout.js"
import configureStore from './store/configureStore.js'

/* ================= Application Components ================= */

const store = configureStore();

store.runSaga();

function MainApp() {

  return (
    <div style={{ margin:0, padding:0, height:'100%' }}>
      <Layout />
    </div>
  );
}

ReactDOM.render(
  <Provider store={store}>
    <MainApp />
  </Provider>,
   
document.getElementById("app"));
