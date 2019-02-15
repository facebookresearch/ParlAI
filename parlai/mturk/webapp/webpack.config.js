/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

var path = require('path');
var webpack = require('webpack');

module.exports = {
  entry: './dev/main.js',
  output: {
    path: __dirname + '/static/',
    filename: '[name].bundle.js',
    publicPath: __dirname + '/static/',
  },
  resolve: {
    extensions: ['.js', '.jsx']
  },
  node: {
    net: 'empty',
    dns: 'empty'
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        loader: 'babel-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        loader: "style-loader!css-loader"
      },
      {
        test: /\.png$/,
        loader: "url-loader?limit=100000"
      },
      {
        test: /\.jpg$/,
        loader: "file-loader"
      },
    ]
  },
};
