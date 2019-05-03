#!/bin/sh

git ls-files parlai/zoo/ tests/nightly/gpu | sort | xargs md5sum
