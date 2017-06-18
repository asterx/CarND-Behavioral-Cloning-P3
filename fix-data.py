#!/usr/bin/env python
import re

re_prefix = re.compile(r'(.*?IMG/(.*?),)')

with open('./data/driving_log.csv') as f:
    lines = f.readlines()

with open('./data/driving_log_fixed.csv', 'w') as f:
    for line in lines:
        line = re.sub(re_prefix, r'IMG/\2,', line)
        f.write(line)
