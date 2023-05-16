#!/bin/bash

set -o errexit

wget_link_file=$1
output_dir=$2

if [ -z ${wget_file} ]; then
  echo "Usage: $0 <wget_file> <output_dir>"
  exit 1
fi

if [ -z ${output_dir} ]; then
  echo "Usage: $0 <wget_file> <output_dir>"
  exit 1
fi

echo "Downloading ${wget_file} into ${output_dir}"
wget -i ${wget_file} -P ${output_dir}