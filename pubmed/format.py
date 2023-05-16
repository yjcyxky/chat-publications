#!/usr/bin/env python

import os
import json
import click

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def write_csv(filepath, data):
    with open(filepath, 'w') as f:
        f.write(data)

def replace_all_chars(text, chars=['\n'], replacement=" "):
    for char in chars:
        text = text.replace(char, replacement)
    return text

@click.command()
@click.option('--filepath', '-f', required=True, help="The path to the json file.")
def main(filepath):
    dirname = os.path.dirname(filepath)
    data = read_json(filepath)

    for row in data:
        csv_content = ''
        keys = list(row.keys())
        pmid = row['pmid']
        for key in keys:
            text = replace_all_chars(str(row[key]))
            csv_content += f"{key}: {text}" + '\n'

        outputfile = os.path.join(dirname, f"{pmid}.txt")
        write_csv(outputfile, csv_content)

if __name__ == '__main__':
    main()