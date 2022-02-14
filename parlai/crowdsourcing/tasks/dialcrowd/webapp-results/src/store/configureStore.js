/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

import {createStore, applyMiddleware, compose} from 'redux'
import createSagaMiddleware from 'redux-saga'
import rootReducer from '../reducers/rootReducer.js'
import rootSaga from '../sagas/rootSaga.js';
import {loadState, saveState} from '../util/localStorage'

const persistedState = loadState();

export default function configureStore() {

  const sagaMiddleware = createSagaMiddleware();
  const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
  const store = createStore(rootReducer, persistedState, composeEnhancers(applyMiddleware(sagaMiddleware)));

  store.subscribe(() => {
    saveState({
      task: store.getState().task,
      crowd: store.getState().crowd
    });
  }, 1000);

  return {
    ...store,
    runSaga: () => sagaMiddleware.run(rootSaga),
  }

}
