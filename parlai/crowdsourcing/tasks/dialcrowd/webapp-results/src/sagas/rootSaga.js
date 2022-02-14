/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import {all, call, put, takeEvery,} from 'redux-saga/effects';

import queryString from 'query-string';

import axios from 'axios';

export default function* rootSaga() {
  yield takeEvery('COM_ALL', combineData);
}

function* combineData(action) {
  yield put({type: 'ALL', all: action.List});
}

