/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

var path = require("path");
var webpack = require("webpack");

module.exports = {
  entry: "./src/main.js",
  output: {
    path: __dirname,
    filename: "build/bundle.js",
  },
  node: {
    net: "empty",
    dns: "empty",
  },
  resolve: {
    alias: {
      react: path.resolve("./node_modules/react"),
    },
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        loader: "babel-loader",
        exclude: /node_modules/,
        options: { presets: ["@babel/env"] },
      },
      {
        test: /\.css$/,
        loader: "style-loader!css-loader",
      },
      {
        test: /\.(svg|png|jpe?g|ttf)$/,
        loader: "url-loader?limit=100000",
      },
      {
        test: /\.jpg$/,
        loader: "file-loader",
      },
    ],
  },
};
