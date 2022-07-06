# Running Tests Locally

Tests are run in ParlAI with [pytest](https://docs.pytest.org/en/stable/).

To run all tests in your current directory, simply run:
```
pytest
```

To run tests from a specific file, run:
```
pytest <filepath>
```

To use name-based filtering to run tests, use the flag `-k`. For example, to only run tests with `TransformerRanker` in the name, run:
```
pytest -k TransformerRanker
```

For verbose testing logs, use `-v`:
```
pytest -v -k TransformerRanker
```


To print the output from a test or set of tests, use `-s`:
```
pytest -s -k TransformerRanker
```