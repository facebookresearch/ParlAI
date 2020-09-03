## Description

`parlai/mturk/core/react_server` contains several files and folders that comprise the server that is built to serve the task into the MTurk UI:

### Folders

- **src/**: contains the react frontend components that comprise the frontend, as well as the css and main javascript file that ties it all together. Of these, the `src/components/core_components.jsx` file is likely the most interesting, as it contains the frontend components that are rendered in a normal task. To replace them, you'll need to create a `custom.jsx` file following the formatting in the dummy version in the same folder. See an example of this in the `react_task_demo` task in the `parlai/mturk/tasks` folder.


### Files

The rest of the files are associated with the process of building the finalized javascript main via node and outputting it into the server directory.

- **.babelrc**: links presets and plugins required for babel to transpile the react jsx files into pure js.
- **package.json**: contains the build dependencies and directs the main build process to run the contents of the webpack config file.
- **webpack.config.js**: configures webpack to grab the contents of the `dev` folder and output the final built file to `server/static`.
