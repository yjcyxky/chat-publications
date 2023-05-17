## Prepare environment

```bash
cd pubmed
virtualenv -p python3 .venv

source .venv/bin/activate
pip install -r requirements.txt
```

## All in one

```bash
bash all-in-one.sh
```

## Step by step
### Make download links

After running `make-download-links.sh`, you can get two files in the current directory.

- `pubmed_links.txt`: download links of all metadata of publications from PubMed.
- `pubmed_baseline.html`: a html file to download all metadata of publications from PubMed.

```bash
cd pubmed
bash make-download-links.sh
```

### Download all metadata of publications from PubMed

we assume that all your data will be stored in `data` directory.

```bash
bash download-pubmed.sh pubmed_links.txt data
```

### Check md5sum

It will take all `.xml.gz` files in `data` directory as input.

```bash
bash check-md5sum.sh data
```

### Extract all metadata of publications and convert it to json format

We assume that all your data will be stored in `data` directory, and all outputed json data will be stored in `data_json` directory. All failed files will be listed in `data_json/failed-list.txt`.

```bash
bash batch-convert.sh data data_json
```

### [Optional] Convert all json file to pubtext format

Why pubtext format? Because we want the llama index can take the whole text of publications as input and treat it as a node.

Pubtext format as follows:
```
pubmed_id: 123456
title: title of publication
abstract: abstract of publication
...
```

```bash
bash json2pubtext.py -f data_json -o data_pubtext
```

### Build index

Please follow the instructions in [README.md](../README.md) to build index.
