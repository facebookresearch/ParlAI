/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
    extensions: ['.js', '.jsx'],
  },
  node: {
    net: 'empty',
    dns: 'empty',
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
        loader: 'style-loader!css-loader',
      },
      {
        test: /\.png$/,
        loader: 'url-loader?limit=100000',
      },
      {
        test: /\.jpg$/,
        loader: 'file-loader',
      },
    ],
  },
};
