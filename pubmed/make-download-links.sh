#!/bin/bash

set -o errexit

echo "Downloading pubmed_baseline.html"
curl https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ > pubmed_baseline.html

files=($(grep -o 'pubmed2[0-9]n[0-9]\{4\}\.xml\.gz' pubmed_baseline.html | sort | uniq))
prefix=$(echo ${files[0]} | cut -d 'n' -f 1)
num_files=${#files[@]}

echo "${num_files}"
echo "Found ${num_files} files with prefix ${prefix}"
echo "Writing links to ${PWD}/pubmed_links.txt"

echo "Initializing pubmed_links.txt"
echo "" > pubmed_links.txt

for ((i=1; i <= ${num_files}; ++i))
do
    num=`printf "%04d\n" ${i}`
    echo "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/${prefix}n${num}.xml.gz.md5" >> pubmed_links.txt
    echo "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/${prefix}n${num}.xml.gz" >> pubmed_links.txt
done

echo "Done"
