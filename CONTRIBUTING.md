# Contributing to ParlAI

While we are seeding this project with an initial set of popular tasks and a few
models and examples, ongoing contributions from the research community are
desired to increase the pool of tasks, models, and baselines.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Make sure your code lints (`bash tests/lint_changed.sh`)
5. Ensure the test suite passes. Run `python setup.py test`.
6. If you've added a new dataset, you should also run
   `python setup.py test -s tests.suites.datatests`. Copy-paste the output into a
   comment in your PR.
7. If you haven't already, complete the Contributor License Agreement ("CLA").

Our continuous integration suite will run only fast tests before merging. We also
have a number of slow, more thorough tests which run nightly. You can additionally
run these by adding `[long]` into any of your commit messages. (A full list of
test triggers may be found in `.circleci/triggers.py`)

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues for general feature discussion, Q&A and public bugs tracking.
Please ensure your description is clear and has sufficient instructions to be able to
reproduce the issue or understand the problem.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
We try to follow the PEP style guidelines and encourage you to as well. You
should run the `lint_changed.sh` script before you submit.

## License
By contributing to ParlAI, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
