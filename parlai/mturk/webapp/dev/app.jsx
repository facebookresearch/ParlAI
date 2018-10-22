import React from 'react';
import ReactDOM from 'react-dom';
import {Button, Panel, Table} from 'react-bootstrap';
import 'fetch';

var AppURLStates = Object.freeze({
  init:0, tasks:1, unsupported:2, runs:3,
});

function convert_time(timestamp){
  var a = new Date(timestamp * 1000);
  var time = (a.toLocaleDateString("en-US") + ' ' +
              a.toLocaleTimeString("en-US"));
  return time;
}

function resolveState(state_string) {
  if (state_string == '') {
    return {url_state: AppURLStates.init, args: null};
  }
  var args = state_string.split('/');
  var state = {
    url_state: AppURLStates.unsupported,
    args: null,
  };
  if (AppURLStates.hasOwnProperty(args[0])) {
    state.url_state = AppURLStates[args[0]];
  }
  if (args.length > 1) {
    state.args = args.slice(1);
  }
  return state;
}

class NavLink extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    if (this.props.type == 'run') {
      return (
        <a href={'/app/runs/' + this.props.target}>
          {this.props.children}
        </a>
      )
    } else {
      return (
        <span>{this.props.children}</span>
      )
    }
  }

}

class TaskTable extends React.Component {
  constructor(props) {
    super(props);
  }

  makeRow(item) {
    return (
      <tr key={item.run_id + '_table_row'}>
        <td>
          <NavLink type='run' target={item.run_id}>
            {item.run_id}
          </NavLink>
        </td>
        <td>complete</td>
        <td>{item.created}</td>
        <td>{item.maximum}</td>
        <td>{item.completed}</td>
        <td>{item.failed}</td>
      </tr>
    )
  }

  render() {
    return (
      <Table striped bordered condensed hover>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Status</th>
            <th>Created</th>
            <th>Maximum</th>
            <th>Completed</th>
            <th>Failed</th>
          </tr>
        </thead>
        <tbody>
          {this.props.items.map(this.makeRow)}
        </tbody>
      </Table>
    )
  }
}

class HitTable extends React.Component {
  constructor(props) {
    super(props);
  }

  makeRow(item) {
    return (
      <tr key={item.hit_id + '_table_row'}>
        <td>
          <NavLink type='hit' target={item.hit_id}>
            {item.hit_id}
          </NavLink>
        </td>
        <td>{convert_time(item.expiration)}</td>
        <td>{item.hit_status}</td>
        <td>{item.assignments_pending}</td>
        <td>{item.assignments_available}</td>
        <td>{item.assignments_complete}</td>
      </tr>
    )
  }

  render() {
    return (
      <Table striped bordered condensed hover>
        <thead>
          <tr>
            <th>HIT ID</th>
            <th>Expiration</th>
            <th>HIT Status</th>
            <th>Pending</th>
            <th>Available</th>
            <th>Complete</th>
          </tr>
        </thead>
        <tbody>
          {this.props.items.map(this.makeRow)}
        </tbody>
      </Table>
    )
  }
}

class AssignmentTable extends React.Component {
  constructor(props) {
    super(props);
  }

  makeRow(item) {
    return (
      <tr key={item.assignment_id + '_table_row'}>
        <td>
          <NavLink type='assignment' target={item.assignment_id}>
            {item.assignment_id}
          </NavLink>
        </td>
        <td>{item.status}</td>
        <td>{convert_time(item.approve_time)}</td>
        <td>
          <NavLink type='worker' target={item.worker_id}>
            {item.worker_id}
          </NavLink>
        </td>
        <td>
          <NavLink type='hit' target={item.hit_id}>
            {item.hit_id}
          </NavLink>
        </td>
      </tr>
    )
  }

  render() {
    return (
      <Table striped bordered condensed hover>
        <thead>
          <tr>
            <th>Assignment ID</th>
            <th>Status</th>
            <th>Approval Time</th>
            <th>Worker ID</th>
            <th>HIT ID</th>
          </tr>
        </thead>
        <tbody>
          {this.props.items.map(this.makeRow)}
        </tbody>
      </Table>
    )
  }
}

class RunPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {runLoading: true, items: null, error: false};
  }

  fetchRunData() {
    fetch('/runs/' + this.props.run_id)
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            runLoading: false,
            data: result
          });
        },
        (error) => {
          this.setState({
            runLoading: false,
            error: true
          });
        }
      )
  }

  componentDidMount() {
    this.setState({runLoading: true});
    this.fetchRunData();
  }

  renderRunInfo() {
    return (
      <div>
        <TaskTable items={[this.state.data.run_details]} />
        <AssignmentTable items={this.state.data.assignments} />
        <HitTable items={this.state.data.hits} />
      </div>
    )
  }

  render() {
    var content;
    if (this.state.runLoading) {
      content = <span>Run details are currently loading...</span>;
    } else if (this.state.error !== false) {
      content = <span>Run loading failed, perhaps run doesn't exist?</span>;
    } else {
      content = this.renderRunInfo();
    }

    return (
      <Panel>
        <Panel.Heading>
          <Panel.Title componentClass="h3">
            Task Details - Run: {this.props.run_id}
          </Panel.Title>
        </Panel.Heading>
        <Panel.Body>
          {content}
        </Panel.Body>
      </Panel>
    )
  }
}

class TaskPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {tasksLoading: true, items: null, error: false};
  }

  fetchTaskData() {
    fetch('/tasks')
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            tasksLoading: false,
            items: result
          });
        },
        (error) => {
          this.setState({
            tasksLoading: false,
            error: error
          });
        }
      )
  }

  componentDidMount() {
    this.setState({tasksLoading: true});
    this.fetchTaskData();
  }

  render() {
    var content;
    if (this.state.tasksLoading) {
      content = <span>Tasks are currently loading...</span>;
    } else if (this.state.error !== false) {
      console.log(this.state.error)
      content = <span>Tasks loading failed...</span>;
    } else {
      content = <TaskTable items={this.state.items}/>;
    }

    return (
      <Panel>
        <Panel.Heading>
          <Panel.Title componentClass="h3">
            Running Tasks List
          </Panel.Title>
        </Panel.Heading>
        <Panel.Body>
          {content}
        </Panel.Body>
      </Panel>
    )
  }
}

class MainApp extends React.Component {
  constructor(props) {
    super(props);
    this.state = resolveState(URL_DESTINATION);
  }

  renderInitPage() {
    return (
      <div>
        <span>Welcome to the ParlAI-Dashboard. Use the button to begin</span>
        <Button
          bsStyle="info"
          href="/app/tasks">
            Click me
        </Button>
      </div>
    );
  }

  renderUnsupportedPage() {
    return (
      <div>
        <span>Oops something happened! use this button to return </span>
        <Button
          bsStyle="info"
          href="/app/tasks">
            Click me
        </Button>
      </div>
    );
  }

  renderTaskPage() {
    return (
      <div style={{width: '600px'}}>
        <TaskPanel/>
      </div>
    );
  }

  renderRunPage() {
    return (
      <div style={{width: '800px'}}>
        <RunPanel run_id={this.state.args[0]}/>
      </div>
    );
  }

  render() {
    if (this.state.url_state == AppURLStates.init) {
      return this.renderInitPage();
    } else if (this.state.url_state == AppURLStates.tasks) {
      return this.renderTaskPage();
    } else if (this.state.url_state == AppURLStates.runs) {
      return this.renderRunPage();
    } else {
      return this.renderUnsupportedPage();
    }
  }
}


var main_app = <MainApp/>;

ReactDOM.render(main_app, document.getElementById('app'));
