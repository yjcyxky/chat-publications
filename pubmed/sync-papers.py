#!/usr/bin/env python3

import sys
import click
import json
import datetime


class DateEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, datetime.datetime):
      return obj.strftime('%Y-%m-%d %H:%M:%S')

    elif isinstance(obj, datetime.date):
      return obj.strftime("%Y-%m-%d")

    else:
      return json.JSONEncoder.default(self, obj)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def fetch_paper(pmid=None, doi=None):
    article = None
    if pmid:
        article = fetch.article_by_pmid(pmid)

    if doi:
        article = fetch.article_by_doi(doi)

    return article


def obj2dict(article):
    map = {
        'pmid': 'pmid',
        'authors': 'authors_str',
        'doi': 'doi',
        'abstract': 'abstract',
        'title': 'title',
        'journal': 'journal',
        'journal_abbr': 'journal',
        'keywords': 'keywords',
        'published_date': 'year'
    }

    return {key: getattr(article, value) for key, value in map.items()}


@click.group()
def single_paper():
    pass


@single_paper.command('single')
@click.option('-p', '--pmid', help='The pmid of a paper.')
@click.option('-d', '--doi', help='The doi number of a paper.')
def paper2dict(pmid, doi):
    """Sync paper's information into prophet database."""
    # For single paper
    from metapub import PubMedFetcher
    fetch = PubMedFetcher()
    if not (input_file or pmid or doi):
        print(bcolors.FAIL)
        print("You need to specified -i/-p/-d.", bcolors.ENDC)
        sys.exit(1)
    else:
        article = fetch_paper(pmid=pmid, doi=doi)
        print(obj2dict(article))


@click.group()
def medline_paper():
    pass


@medline_paper.command('medline')
@click.option('-i', '--xml-file', help='Medline XML file.')
@click.option('-o',
              '--output-file',
              help='Output file.',
              default="medline.json")
def parse_medline_xml(xml_file, output_file):
    """Import medline XML file into prophet database."""
    # For medline
    import pubmed_parser as pp
    dicts_out = pp.parse_medline_xml(xml_file,
                                     year_info_only=False,
                                     nlm_category=False,
                                     author_list=False,
                                     reference_list=False)

    with open(output_file, 'w') as fp:
        json.dump(dicts_out, fp, cls=DateEncoder)


@click.group()
def oa_paper():
    pass


@oa_paper.command('oa')
@click.option('-i', '--xml-file', help='Medline XML file.')
@click.option('-m', '--mode', help='What need to parse?',
              type=click.Choice(['paper', 'paragraphs', 'references', 'tables', 'figures']))
@click.option('-o',
              '--output-file',
              help='Output file.',
              default="medline.json")
def parse_oa_xml(xml_file, output_file, mode):
    """Import pubmed open access XML file into prophet database."""
    # For open access
    import pubmed_parser as pp

    if mode == 'paper':
        dicts_out = pp.parse_pubmed_xml(xml_file)
    elif mode == 'paragraphs':
        dicts_out = pp.parse_pubmed_paragraph(xml_file, all_paragraph=True)
    elif mode == 'references':
        dicts_out = pp.parse_pubmed_references(xml_file)
    elif mode == 'tables':
        dicts_out = pp.parse_pubmed_table(xml_file, return_xml=False)
    elif mode == 'figures':
        dicts_out = pp.parse_pubmed_caption(xml_file)

    with open(output_file, 'w') as fp:
        json.dump(dicts_out, fp, cls=DateEncoder)


cli = click.CommandCollection(sources=[single_paper, medline_paper, oa_paper])

if __name__ == '__main__':
    cli()
