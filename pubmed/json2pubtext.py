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
@click.option('--filepath', '-f', required=True, help="The path to the json file(s).")
@click.option('--output', '-o', required=True, help="The path to the output file.", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(filepath, output):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File/Directory not found: {filepath}")
        
    def process_file(filepath, outputdir):
        data = read_json(filepath)

        for row in data:
            csv_content = ''
            keys = list(row.keys())
            pmid = row['pmid']
            for key in keys:
                text = replace_all_chars(str(row[key]))
                csv_content += f"{key}: {text}" + '\n'

            outputfile = os.path.join(outputdir, f"{pmid}.txt")
            write_csv(outputfile, csv_content)

    if os.path.isdir(filepath):
        for file in os.listdir(filepath):
            if file.endswith('.json'):
                file = os.path.join(filepath, file)
                process_file(file, outputdir=output)
    else:
        process_file(filepath, outputdir=output)


if __name__ == '__main__':
    main()