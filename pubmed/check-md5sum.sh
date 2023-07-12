#!/bin/bash

set -o errexit

dir=$1

if [ -z ${dir} ]; then
  echo "Usage: $0 <dir>"
  exit 1
fi

allfiles=(`ls ${dir}/*.gz`)

for i in "${allfiles[@]}"
do
  md5=`md5sum ${i} | cut -d ' ' -f 1`
  source_md5=`cat ${i}.md5 | cut -d ' ' -f 2`

  if [ ${source_md5} != ${md5} ]; then
    echo "${i}: not matched"
  else
    echo "${i}: matched"
  fi
done