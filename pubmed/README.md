## Make download links

After running `make-download-links.sh`, you can get two files in the current directory.

- `pubmed_links.txt`: download links of all metadata of publications from PubMed.
- `pubmed_baseline.html`: a html file to download all metadata of publications from PubMed.

```bash
cd pubmed
bash make-download-links.sh
```

## Download all metadata of publications from PubMed

```bash
bash download-pubmed.sh pubmed_links.txt data
```

## Check md5sum

```bash
bash check-md5sum.sh
```