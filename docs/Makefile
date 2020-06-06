# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going  # make sure we fail on warnings
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ParlAI
SOURCEDIR     = source
BUILDDIR      = build


# Put it first so that "make" without argument is like "make html".
all:
	make html

.PHONY: all Makefile

clean:
	rm -rfv "${BUILDDIR}"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	rm -f source/agent_refs/*
	cd "${SOURCEDIR}"; python generate_agent_list.py
	cd "${SOURCEDIR}"; python generate_task_list.py
	cd "${SOURCEDIR}"; python generate_zoo_list.py
	cd "${SOURCEDIR}"; python generate_cli.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
