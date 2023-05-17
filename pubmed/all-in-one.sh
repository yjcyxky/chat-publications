#!/bin/bash

echo "Make download links"
bash make-download-links.sh

echo "Download files"
mkdir data
bash download-pubmed.sh pubmed_links.txt data

echo "Check files..."
bash check-md5sum.sh data

echo "Convert XML to JSON"
mkdir data_json
bash batch-convert.sh data data_json

# echo "Convert all JSON files to one pubtext file"
# mkdir data_pubtext
# bash json2pubtext.py -f data_json -o data_pubtext

echo "Please follow the instructions in ../README.md to build index."