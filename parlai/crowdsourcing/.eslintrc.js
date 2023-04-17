module.exports = {
  env: {
    browser: true,
    node: true,
    es2021: true
  },
  extends: ["eslint:recommended", "plugin:react/recommended"],
  overrides: [],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module"
  },
  rules: {
    "react/prop-types": 0
  }
};
