import React from 'react';
import ReactDOM from 'react-dom';
import {Jumbotron} from 'react-bootstrap';
import CustomComponents from './components/custom.jsx'
import 'fetch';


class MainApp extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <Jumbotron>
        <h1>It worked!</h1>
      </Jumbotron>
    );
  }
}

if (CustomComponents.MainApp !== undefined) {
  MainApp = CustomComponents.MainApp;
}

var main_app = <MainApp/>;

ReactDOM.render(main_app, document.getElementById('app'));
