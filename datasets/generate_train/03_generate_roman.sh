#!/bin/bash

# python3 generate_roman.py --lang ja --rewrite
# python3 generate_roman.py --lang zh --rewrite
python3 generate_roman.py --lang en-ja --rewrite --trans_type science_trans
python3 generate_roman.py --lang en-zh --rewrite --trans_type science_trans
python3 generate_roman.py --lang zh-ja --rewrite --trans_type science_trans
