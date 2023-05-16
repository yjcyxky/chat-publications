#!/bin/bash

datadir=$0
outputdir=$1
scriptdir=`dirname $0`

if [ -z "$datadir" ]; then
  echo "Usage: batch-convert.sh <datadir> <outputdir>"
  exit 1
fi

if [ -z "$outputdir" ]; then
  echo "Usage: batch-convert.sh <datadir> <outputdir>"
  exit 1
fi

if [ ! -d "$datadir" ]; then
  echo "Directory $datadir does not exist"
  exit 1
fi

if [ ! -d "$outputdir" ]; then
  echo "Directory $outputdir does not exist"
  exit 1
fi

allfiles=(`ls ${datadir}/*.gz`)

for i in "${allfiles[@]}"
do
  filename=`echo $(basename "${i}") | cut -d '.' -f 1`
  python3 ${scriptdir}/sync-papers.py medline -i ${i} -o ${outputdir}/${filename}.json

  if [ "$?" != 0 ]; then
    echo "Failed to convert $i"
    echo "$i" >> ${outputdir}/failed-list.txt
  else
    echo "Convert $i: success"
  fi
done
