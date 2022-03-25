#!/bin/bash
work_dir=$(dirname $(realpath $0))
cd $work_dir
rm -rf train/result train/Blendmask
rm -rf capture/img
rm -rf annotation/labels
