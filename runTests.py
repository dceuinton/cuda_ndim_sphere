#!/usr/bin/env python

import subprocess, os

dir_path = os.path.dirname(os.path.realpath(__file__))
ex = os.path.join(dir_path, "main")

for i in range(27):
	subprocess.call([ex, "%d" % i])
