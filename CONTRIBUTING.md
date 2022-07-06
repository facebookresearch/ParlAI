# Contributing to ParlAI

While we are seeding this project with an initial set of popular tasks and a few
models and examples, ongoing contributions from the research community are
desired to increase the pool of tasks, models, and baselines.


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and then clone the forked repository. (See this [github guide](https://guides.github.com/activities/forking/) on forking for more info).
   **If you have already cloned the repo directly and committed changes, follow the steps in the [section below](#moving-changes-youve-committed-to-a-fork)**
2. Create your branch from `main`. Set up your environment
   and run `pre-commit install` once.
3. Make your changes
4. If you've added code that should be tested, [add tests](http://parl.ai/docs/tutorial_tests.html).
5. If you've changed APIs, update the documentation.
6. Autoformat and lint your code (`bash autoformat.sh`)
7. (Optional) Ensure the test suite passes. Run `python -m pytest -m unit`.
8. If you've added a new dataset, you should also run
   `python -m pytest -m data`. Copy-paste the output into a comment in your PR.
9. If you haven't already, complete the Contributor License Agreement ("CLA").
10. Link [CircleCI](https://circleci.com/vcs-authorize/) to your github account
   if you haven't done so previously (and make sure the CircleCI tests run
   successfully on the PR after you push your changes).
11. Push your changes!
12. Once the PR is accepted and CI is passing, we will merge the PR for you.

### Moving changes you've committed to a fork
1. Fork the repo
2. In your local repo, rename your origin remote to upstream
   ```
   git remote rename origin upstream
   ```
3. Point origin to the forked repo (instead of to the original repo)
   ```
   git remote add origin git@github...<FORK>
   ```
4. Fetch from the new origin
   ```
   git fetch origin
   ```
5. Make your local branch track the remote branch (of the forked repo)
   ```
   git branch --set-upstream-to origin/main main
   ```

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
