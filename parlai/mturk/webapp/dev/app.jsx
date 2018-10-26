import React from 'react';
import ReactDOM from 'react-dom';
import {Button, Panel, Table} from 'react-bootstrap';
import ReactTable from "react-table";
import 'react-table/react-table.css';
import 'fetch';

var AppURLStates = Object.freeze({
  init:0, tasks:1, unsupported:2, runs:3, workers:4,
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
      );
    } else if (this.props.type == 'worker') {
      return (
        <a href={'/app/workers/' + this.props.target}>
          {this.props.children}
        </a>
      );
    } else {
      return (
        <span>{this.props.children}</span>
      )
    }
  }

}

class SharedTable extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    var used_cols = this.props.used_cols;
    var row_formatter = used_cols.map(this.props.getColumnFormatter, this.props);
    return (
      <Panel id={this.props.title} defaultExpanded>
        <Panel.Heading>
          <Panel.Title componentClass="h3">
            {this.props.title}
          </Panel.Title>
        </Panel.Heading>
        <Panel.Body style={{padding: '0px'}}>
          <ReactTable
            className='-striped -highlight'
            showPagination={this.props.data.length > 20}
            sortable={this.props.data.length > 1}
            filterable={false}
            minRows="1"
            columns={row_formatter}
            {...this.props}
          />
        </Panel.Body>
      </Panel>
    )
  }
}

class TaskTable extends React.Component {
  constructor(props) {
    super(props);
    this.state = {used_cols: [
      'run_id', 'run_status', 'created', 'maximum', 'completed', 'failed',
    ]};
  }

  getColumnFormatter(row_name) {
    return {
      id: row_name,
      Header: props => this.getHeaderValue(row_name),
      accessor: item => this.getColumnValue(row_name, item),
      Cell: props => this.getColumnCell(row_name, props),
    };
  }

  getHeaderValue(header_name) {
    switch(header_name) {
      case 'run_id':
        return <span>Run ID</span>;
      case 'run_status':
        return <span>Status</span>;
      case 'created':
        return <span>Created</span>;
      case 'maximum':
        return <span>Maximum</span>;
      case 'completed':
        return <span>Completed</span>;
      case 'failed':
        return <span>Failed</span>;
      default:
        return <span>Invalid column {header_name}</span>;
    }
  }

  getColumnCell(header_name, props) {
    // TODO add table row/icon for `notes` that appear on hover
    switch(header_name) {
      case 'run_id':
        return <NavLink type='run' target={props.row.run_id}>
          {props.value}
        </NavLink>;
      case 'run_status':
      case 'created':
      case 'maximum':
      case 'completed':
      case 'failed':
      default:
        return <span>{props.value}</span>;
    }
  }

  getColumnValue(header_name, item) {
    switch(header_name) {
      case 'run_id': return item.run_id;
      case 'run_status': return item.run_status;
      case 'created': return item.created;
      case 'maximum': return item.maximum;
      case 'completed': return item.completed;
      case 'failed': return item.failed;
      default: return 'Invalid column ' + header_name;
    }
  }

  render() {
    return (
      <SharedTable
        getColumnFormatter={this.getColumnFormatter.bind(this)}
        used_cols={this.state.used_cols}
        data={this.props.data}
        title={this.props.title}
      />
    );
  }
}

class HitTable extends React.Component {
  constructor(props) {
    super(props);
    this.state = {used_cols: [
      'hit_id', 'expiration', 'hit_status', 'assignments_pending',
      'assignments_available', 'assignments_complete',
    ]};
  }

  getColumnFormatter(row_name) {
    return {
      id: row_name,
      Header: props => this.getHeaderValue(row_name),
      accessor: item => this.getColumnValue(row_name, item),
      Cell: props => this.getColumnCell(row_name, props),
    };
  }

  getHeaderValue(header_name) {
    switch(header_name) {
      case 'hit_id':
        return <span>HIT ID</span>;
      case 'expiration':
        return <span>Expiration</span>;
      case 'hit_status':
        return <span>HIT Status</span>;
      case 'assignments_pending':
        return <span>Pending</span>;
      case 'assignments_available':
        return <span>Available</span>;
      case 'assignments_complete':
        return <span>Complete</span>;
      default:
        return <span>Invalid column {header_name}</span>;
    }
  }

  getColumnCell(header_name, props) {
    // TODO add table row/icon for `notes` that appear on hover
    switch(header_name) {
      case 'hit_id':
        return <NavLink type='hit' target={props.row.hit_id}>
          {props.value}
        </NavLink>;
      case 'expiration':
      case 'hit_status':
      case 'assignments_pending':
      case 'assignments_available':
      case 'assignments_complete':
      default:
        return <span>{props.value}</span>;
    }
  }

  getColumnValue(header_name, item) {
    switch(header_name) {
      case 'hit_id': return item.hit_id;
      case 'expiration':
        if (item.expiration > 0) {
          return convert_time(item.expiration);
        } else {
          return 'No expiration recorded';
        }
      case 'hit_status': return item.hit_status;
      case 'assignments_pending': return item.assignments_pending;
      case 'assignments_available': return item.assignments_available;
      case 'assignments_complete': return item.assignments_complete;
      default: return 'Invalid column ' + header_name;
    }
  }

  render() {
    return (
      <SharedTable
        getColumnFormatter={this.getColumnFormatter.bind(this)}
        used_cols={this.state.used_cols}
        data={this.props.data}
        title={this.props.title}
      />
    );
  }
}

class AssignmentTable extends React.Component {
  constructor(props) {
    super(props);
    // TODO additional rows conversation_id, bonus_text,
    this.state = {used_cols: [
      'assignment_id', 'worker_id', 'task_status', 'world_status',
      'approve_time', 'onboarding_start', 'onboarding_end', 'task_start',
      'task_end', 'bonus_amount', 'bonus_paid', 'hit_id', 'run_id',
    ]};
  }

  getColumnFormatter(row_name) {
    return {
      id: row_name,
      Header: props => this.getHeaderValue(row_name),
      accessor: item => this.getColumnValue(row_name, item),
      Cell: props => this.getColumnCell(row_name, props),
    };
  }

  getHeaderValue(header_name) {
    switch(header_name) {
      case 'assignment_id':
        return <span>Assignment ID</span>;
      case 'worker_id':
        return <span>Worker ID</span>;
      case 'task_status':
        return <span>Task Status</span>;
      case 'world_status':
        return <span>World Status</span>;
      case 'approve_time':
        return <span>Approve By</span>;
      case 'onboarding_start':
        return <span>Onboarding Start</span>;
      case 'onboarding_end':
        return <span>Onboarding End</span>;
      case 'task_start':
        return <span>Task Start</span>;
      case 'task_end':
        return <span>Task End</span>;
      case 'bonus_amount':
        return <span>Bonus Amount</span>;
      case 'bonus_paid':
        return <span>Bonus Paid</span>;
      case 'hit_id':
        return <span>HIT ID</span>;
      case 'run_id':
        return <span>Run ID</span>;
      default:
        return <span>Invalid column {header_name}</span>;
    }
  }

  getColumnCell(header_name, props) {
    // TODO add table row/icon for `notes` that appear on hover
    switch(header_name) {
      case 'assignment_id':
        return <NavLink type='assignment' target={props.row.assignment_id}>
          {props.value}
        </NavLink>;
      case 'worker_id':
        return <NavLink type='worker' target={props.row.worker_id}>
          {props.value}
        </NavLink>;
      case 'hit_id':
        return <NavLink type='hit' target={props.row.hit_id}>
          {props.value}
        </NavLink>;
      case 'run_id':
        return <NavLink type='run' target={props.row.run_id}>
          {props.value}
        </NavLink>;
      case 'task_status':
      case 'world_status':
      case 'approve_time':
      case 'onboarding_start':
      case 'onboarding_end':
      case 'task_start':
      case 'task_end':
      case 'bonus_amount':
      case 'bonus_paid':
      default:
        return <span>{props.value}</span>;
    }
  }

  getColumnValue(header_name, item) {
    switch(header_name) {
      case 'assignment_id': return item.assignment_id;
      case 'worker_id': return item.worker_id;
      case 'task_status': return item.status;
      case 'world_status': return item.world_status;
      case 'approve_time':
        if (item.approve_time > 0) {
          return convert_time(item.approve_time);
        } else {
          return 'Not completed';
        }
      case 'onboarding_start':
        if (item.onboarding_start > 0) {
          return convert_time(item.onboarding_start);
        } else {
          return 'No onboarding';
        }
      case 'onboarding_end':
        if (item.onboarding_end > 0) {
          return convert_time(item.onboarding_end);
        } else {
          return' Never entered task queue';
        }
      case 'task_start':
        if (item.task_start > 0) {
          return convert_time(item.task_start);
        } else {
          return 'Never started task';
        }
      case 'task_end':
        if (item.task_end > 0) {
          return convert_time(item.task_end);
        } else {
          return 'Never finished task';
        }
      case 'bonus_amount': return item.bonus_amount;
      case 'bonus_paid': return item.bonus_paid;
      case 'hit_id': return item.hit_id;
      case 'run_id': return item.run_id;
      default: return 'Invalid column ' + header_name;
    }
  }

  render() {
    return (
      <SharedTable
        getColumnFormatter={this.getColumnFormatter.bind(this)}
        used_cols={this.state.used_cols}
        data={this.props.data}
        title={this.props.title}
      />
    );
  }
}

class WorkerTable extends React.Component {
  constructor(props) {
    super(props);
    // TODO additional rows conversation_id, bonus_text,
    this.state = {used_cols: [
      'worker_id', 'accepted', 'disconnected', 'completed',
      'approved', 'rejected',
    ]};
  }

  getColumnFormatter(row_name) {
    return {
      id: row_name,
      Header: props => this.getHeaderValue(row_name),
      accessor: item => this.getColumnValue(row_name, item),
      Cell: props => this.getColumnCell(row_name, props),
    };
  }

  getHeaderValue(header_name) {
    switch(header_name) {
      case 'worker_id':
        return <span>Worker ID</span>;
      case 'accepted':
        return <span>Accepted</span>;
      case 'disconnected':
        return <span>Disconnected</span>;
      case 'completed':
        return <span>Completed</span>;
      case 'approved':
        return <span>Approved</span>;
      case 'rejected':
        return <span>Rejected</span>;
      default:
        return <span>Invalid column {header_name}</span>;
    }
  }

  getColumnCell(header_name, props) {
    switch(header_name) {
      case 'worker_id':
        return <NavLink type='worker' target={props.row.worker_id}>
          {props.value}
        </NavLink>;
      case 'accepted':
      case 'disconnected':
      case 'completed':
      case 'approved':
      case 'rejected':
      default:
        return <span>{props.value}</span>;
    }
  }

  getColumnValue(header_name, item) {
    switch(header_name) {
      case 'worker_id': return item.worker_id;
      case 'accepted': return item.accepted;
      case 'disconnected': return item.disconnected;
      case 'completed': return item.completed;
      case 'approved': return item.approved;
      case 'rejected': return item.rejected;
      default: return 'Invalid column ' + header_name;
    }
  }

  render() {
    return (
      <SharedTable
        getColumnFormatter={this.getColumnFormatter.bind(this)}
        used_cols={this.state.used_cols}
        data={this.props.data}
        title={this.props.title}
      />
    );
  }
}

class RunPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {run_loading: true, items: null, error: false};
  }

  fetchRunData() {
    fetch('/runs/' + this.props.run_id)
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            run_loading: false,
            data: result
          });
        },
        (error) => {
          this.setState({
            run_loading: false,
            error: true
          });
        }
      )
  }

  componentDidMount() {
    this.setState({run_loading: true});
    this.fetchRunData();
  }

  renderRunInfo() {
    return (
      <div>
        <TaskTable
          data={[this.state.data.run_details]}
          title={'Baseline info for this run'}/>
        <AssignmentTable
          data={this.state.data.assignments}
          title={'Assignments from this run'}/>
        <HitTable
          data={this.state.data.hits}
          title={'HITs from this run'}/>
      </div>
    )
  }

  render() {
    var content;
    if (this.state.run_loading) {
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

class WorkerPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {worker_loading: true, items: null, error: false};
  }

  fetchRunData() {
    fetch('/workers/' + this.props.worker_id)
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            worker_loading: false,
            data: result
          });
        },
        (error) => {
          this.setState({
            worker_loading: false,
            error: true
          });
        }
      )
  }

  componentDidMount() {
    this.setState({worker_loading: true});
    this.fetchRunData();
  }

  renderRunInfo() {
    return (
      <div>
        <WorkerTable
          data={[this.state.data.worker_details]}
          title={'Worker Stats'}/>
        <AssignmentTable
          data={this.state.data.assignments}
          title={'Assignments from this Worker'}/>
      </div>
    )
  }

  render() {
    var content;
    if (this.state.worker_loading) {
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
            Worker Details - Worker: {this.props.worker_id}
          </Panel.Title>
        </Panel.Heading>
        <Panel.Body>
          {content}
        </Panel.Body>
      </Panel>
    )
  }
}

class TaskListPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {tasks_loading: true, items: null, error: false};
  }

  fetchTaskData() {
    fetch('/tasks')
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            tasks_loading: false,
            items: result
          });
        },
        (error) => {
          this.setState({
            tasks_loading: false,
            error: error
          });
        }
      )
  }

  componentDidMount() {
    this.setState({tasks_loading: true});
    this.fetchTaskData();
  }

  render() {
    var content;
    if (this.state.tasks_loading) {
      content = <span>Tasks are currently loading...</span>;
    } else if (this.state.error !== false) {
      console.log(this.state.error)
      content = <span>Tasks loading failed...</span>;
    } else {
      content = <TaskTable data={this.state.items} title={'Local Runs'}/>;
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

class WorkerListPanel extends React.Component {
  constructor(props) {
    super(props);
    this.state = {workers_loading: true, items: null, error: false};
  }

  fetchTaskData() {
    fetch('/workers')
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({
            workers_loading: false,
            items: result
          });
        },
        (error) => {
          this.setState({
            workers_loading: false,
            error: error
          });
        }
      )
  }

  componentDidMount() {
    this.setState({workers_loading: true});
    this.fetchTaskData();
  }

  render() {
    var content;
    if (this.state.workers_loading) {
      content = <span>Workers are currently loading...</span>;
    } else if (this.state.error !== false) {
      console.log(this.state.error)
      content = <span>Workers loading failed...</span>;
    } else {
      content = (
        <WorkerTable
          data={this.state.items}
          title={'Workers'}/>
      );
    }

    return (
      <Panel>
        <Panel.Heading>
          <Panel.Title componentClass="h3">
            Workers List
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
      <div style={{width: '800px'}}>
        <TaskListPanel/>
        <WorkerListPanel/>
      </div>
    );
  }

  renderWorkerPage() {
    return (
      <div style={{width: '800px'}}>
        <WorkerPanel worker_id={this.state.args[0]}/>
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
    } else if (this.state.url_state == AppURLStates.workers) {
      return this.renderWorkerPage();
    } else {
      return this.renderUnsupportedPage();
    }
  }
}


var main_app = <MainApp/>;

ReactDOM.render(main_app, document.getElementById('app'));
