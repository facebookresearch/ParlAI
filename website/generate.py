#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Builds the ParlAI website.
"""

import os
import git
import markdown
from mdx_gfm import PartialGithubFlavoredMarkdownExtension

GIT_ROOT_LEVEL = git.Git().rev_parse('--show-toplevel')
WEBSITE_ROOT = os.path.join(GIT_ROOT_LEVEL, 'website')
TEMPLATES = os.path.join(WEBSITE_ROOT, 'templates')
OUT_DIR = os.path.join(WEBSITE_ROOT, 'build')


def ghmarkdown(source):
    return markdown.markdown(
        source,
        extensions=[PartialGithubFlavoredMarkdownExtension()]
    )


def _read_file(filename):
    with open(filename) as f:
        return f.read()


def _mkdirp(directory):
    """Equivalent to mkdir -p"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def _write_file(partial_filename, content):
    filename = os.path.join(OUT_DIR, partial_filename)

    _mkdirp(os.path.dirname(filename))

    print("writing {} bytes to {}".format(len(content), filename))
    with open(filename, 'w') as f:
        f.write(content)


def wrap_base(content, title):
    template = _read_file(os.path.join(TEMPLATES, 'base.html'))
    template = template.replace('{{{CONTENT}}}', content)
    template = template.replace('{{{TITLE}}}', title)
    return template


def make_errorpage():
    content = _read_file(os.path.join(TEMPLATES, 'error.html'))
    html = wrap_base(content, "Error")
    _write_file('error.html', html)


def make_aboutpage():
    template = _read_file(os.path.join(TEMPLATES, 'about.html'))
    readme = _read_file(os.path.join(GIT_ROOT_LEVEL, 'README.md'))
    # filter out the circleci badge from the about page
    readme = "\n".join([
        l for l in readme.split("\n")
        if not l.startswith("[![CircleCI]")
    ])
    readme_html = ghmarkdown(readme)
    readme_html = readme_html.replace("docs/source/\\", "/docs/")
    content = template.replace('{{{CONTENT}}}', readme_html)
    html = wrap_base(content, "About | ParlAI")
    _write_file('about/index.html', html)


def make_homepage():
    template = _read_file(os.path.join(TEMPLATES, 'home.html'))
    news = _read_file(os.path.join(GIT_ROOT_LEVEL, 'NEWS.md'))
    news = news.replace('## News', '')
    news_html = ghmarkdown(news)
    content = template.replace('{{{CONTENT}}}', news_html)
    html = wrap_base(content, "ParlAI")
    _write_file('index.html', html)


def make_projects_landing():
    template = _read_file(os.path.join(TEMPLATES, 'project.html'))
    landing = _read_file(os.path.join(GIT_ROOT_LEVEL, 'projects/README.md'))
    landing = landing.replace(
        'This directory also contains subfolders for some of the projects which are '
        'housed in the ParlAI repo, others are maintained via external websites. '
        'Please also refer',
        'See the [ParlAI projects](https://github.com/facebookresearch/ParlAI/'
        'tree/master/projects) page on GitHub for more information. Refer'
    )
    landing_html = template.replace('{{{CONTENT}}}', ghmarkdown(landing))
    html = wrap_base(landing_html, "Projects | ParlAI")
    _write_file('projects/index.html', html)


def make_projects_individual():
    template = _read_file(os.path.join(TEMPLATES, 'project.html'))
    projects_dir = os.path.join(GIT_ROOT_LEVEL, 'projects')
    possible_projects = os.listdir(projects_dir)
    projects = [
        pp for pp in possible_projects
        if os.path.exists(os.path.join(projects_dir, pp, 'README.md'))
    ]
    for p in projects:
        content = _read_file(os.path.join(projects_dir, p, 'README.md'))
        content_html = template.replace('{{{CONTENT}}}', ghmarkdown(content))
        content_html = content_html.replace(
            'src="',
            'src="https://raw.githubusercontent.com/facebookresearch/'
            'ParlAI/master/projects/{}'
            .format(p + '/' if p else '')
        )
        title = p.title().replace("_", " ")
        html = wrap_base(content_html, title)
        _write_file(os.path.join('projects', p, 'index.html'), html)


def main():
    make_errorpage()
    make_homepage()
    make_aboutpage()
    make_projects_landing()
    make_projects_individual()


if __name__ == '__main__':
    main()
