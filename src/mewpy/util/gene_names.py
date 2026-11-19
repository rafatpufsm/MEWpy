"""
Mapping of gene and protein names to locus tags.

Literature and text-mining tools name genes by symbol ("arcA"), protein name ("ArcA") or an old
synonym ("dye", "sfrA"), whereas genome-scale models identify genes by locus tag, e.g. the
Blattner b-number in E. coli ("b4401") or the Rv number in M. tuberculosis ("Rv0001").
:class:`GeneNameMapper` resolves the former into the latter.

This is a Python reimplementation of the identifier normalisation done by the KREN repository
(R. T. Pereira, PhD thesis, University of Minho, 2016), which used the b-number as the common
key across NCBI, EcoCyc, KEGG and RegulonDB.

Name sources are ranked so that an official symbol wins over a synonym of another gene:

    0 - the locus tag itself
    1 - official gene symbol
    2 - synonym / alias

Matching ignores case. A name that remains ambiguous at its best rank is not resolved;
:meth:`GeneNameMapper.candidates` lists the competing locus tags.

Example:

    mapper = GeneNameMapper.from_ncbi_gene_info('Escherichia_coli_str._K-12_substr._MG1655.gene_info.gz')
    mapper.resolve('ArcA')   # 'b4401'
    mapper.resolve('sfrA')   # 'b4401'
"""
import csv
import gzip
import os
import urllib.request
from collections import defaultdict
from typing import Dict, Iterable, Optional, Set, Tuple, Union

LOCUS_TAG = 0
SYMBOL = 1
SYNONYM = 2

NCBI_GENE_INFO_URL = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/{division}/{organism}.gene_info.gz'
KEGG_LIST_URL = 'https://rest.kegg.jp/list/{organism}'

PathLike = Union[str, 'os.PathLike']


def normalize_name(name: str) -> str:
    """
    Normalises a gene or protein name for matching: surrounding whitespace and quotes are removed
    and the name is lower-cased, so 'ArcA', 'arcA' and ' ARCA ' are the same key.
    """
    return name.strip().strip('"\'').strip().lower()


def _open_text(path: PathLike):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')


class GeneNameMapper:
    """
    Resolves gene and protein names into locus tags (e.g. E. coli b-numbers).
    """

    def __init__(self):
        # normalized name -> {locus tag: best rank of that name for that tag}
        self._index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._symbols: Dict[str, str] = {}
        # locus tag -> names as written by the source (kept for to_csv)
        self._names: Dict[str, Set[str]] = defaultdict(set)

    def __len__(self):
        return len(self._symbols)

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) is not None

    @property
    def locus_tags(self) -> Set[str]:
        """The locus tags known by the mapper."""
        return set(self._symbols)

    def symbol(self, locus_tag: str) -> Optional[str]:
        """The official symbol of a locus tag, if known."""
        return self._symbols.get(locus_tag)

    def add(self, locus_tag: str, symbol: Optional[str] = None, synonyms: Iterable[str] = ()):
        """
        Adds a gene to the mapper.

        :param locus_tag: the identifier used by the model (e.g. 'b4401')
        :param symbol: the official gene symbol (e.g. 'arcA')
        :param synonyms: other names of the gene or its protein
        """
        locus_tag = locus_tag.strip()
        if not locus_tag or locus_tag == '-':
            return

        self._register(locus_tag, locus_tag, LOCUS_TAG)

        if symbol and symbol != '-':
            self._register(symbol, locus_tag, SYMBOL)
            self._symbols.setdefault(locus_tag, symbol.strip())
        else:
            self._symbols.setdefault(locus_tag, locus_tag)

        for synonym in synonyms:
            if synonym and synonym != '-':
                self._register(synonym, locus_tag, SYNONYM)

    def _register(self, name: str, locus_tag: str, rank: int):
        key = normalize_name(name)
        if not key:
            return
        tags = self._index[key]
        tags[locus_tag] = min(rank, tags.get(locus_tag, rank))
        self._names[locus_tag].add(name.strip())

    def candidates(self, name: str) -> Dict[str, int]:
        """
        All locus tags a name may refer to, with the rank of the match
        (0 locus tag, 1 symbol, 2 synonym).
        """
        return dict(self._index.get(normalize_name(name), {}))

    def resolve(self, name: str) -> Optional[str]:
        """
        Resolves a name into a single locus tag.

        :param name: a gene symbol, protein name, synonym or locus tag
        :return: the locus tag, or None when the name is unknown or ambiguous
        """
        tags = self.candidates(name)
        if not tags:
            return None

        best = min(tags.values())
        winners = [tag for tag, rank in tags.items() if rank == best]
        if len(winners) == 1:
            return winners[0]
        return None

    def resolve_many(self, names: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Resolves several names at once.

        :return: a dictionary name -> locus tag for the resolved names and a dictionary
        name -> candidates for the names that were unknown (empty candidates) or ambiguous
        """
        resolved, unresolved = {}, {}
        for name in names:
            tag = self.resolve(name)
            if tag is None:
                unresolved[name] = self.candidates(name)
            else:
                resolved[name] = tag
        return resolved, unresolved

    def update(self, other: 'GeneNameMapper'):
        """Merges the names of another mapper into this one."""
        for key, tags in other._index.items():
            for tag, rank in tags.items():
                current = self._index[key]
                current[tag] = min(rank, current.get(tag, rank))
        for tag, symbol in other._symbols.items():
            self._symbols.setdefault(tag, symbol)
        for tag, names in other._names.items():
            self._names[tag].update(names)
        return self

    # -----------------------------------------------------------------------------
    # Readers
    # -----------------------------------------------------------------------------
    @classmethod
    def from_ncbi_gene_info(cls, path: PathLike) -> 'GeneNameMapper':
        """
        Builds a mapper from an NCBI gene_info file (plain or .gz), e.g.
        Escherichia_coli_str._K-12_substr._MG1655.gene_info.gz.
        Uses the LocusTag column as identifier, Symbol and
        Symbol_from_nomenclature_authority as symbols and Synonyms as synonyms.
        Genes without a locus tag are skipped.
        """
        mapper = cls()
        with _open_text(path) as handle:
            header = None
            for line in handle:
                fields = line.rstrip('\n').split('\t')
                if line.startswith('#'):
                    header = [field.lstrip('#') for field in fields]
                    continue
                if header is None:
                    raise ValueError(f'{path} is not an NCBI gene_info file: missing the #tax_id header')

                row = dict(zip(header, fields))
                synonyms = row.get('Synonyms', '-').split('|')
                authority = row.get('Symbol_from_nomenclature_authority', '-')
                mapper.add(row.get('LocusTag', '-'), row.get('Symbol'), synonyms)
                if authority not in ('-', row.get('Symbol')):
                    mapper._register(authority, row.get('LocusTag', '-').strip(), SYMBOL)
        return mapper

    @classmethod
    def from_kegg_list(cls, path: PathLike) -> 'GeneNameMapper':
        """
        Builds a mapper from the output of the KEGG REST list operation for an organism
        (https://rest.kegg.jp/list/eco), whose lines look like
        'eco:b4401<TAB>CDS<TAB>position<TAB>arcA, cpxC; description'.
        The first name is taken as the symbol and the remaining ones as synonyms.
        """
        mapper = cls()
        with _open_text(path) as handle:
            for line in handle:
                fields = line.rstrip('\n').split('\t')
                if len(fields) < 2:
                    continue
                locus_tag = fields[0].split(':', 1)[-1]
                names = fields[-1].split(';', 1)[0] if ';' in fields[-1] else ''
                names = [name.strip() for name in names.split(',') if name.strip()]
                symbol = names[0] if names else None
                mapper.add(locus_tag, symbol, names[1:])
        return mapper

    @classmethod
    def from_csv(cls, path: PathLike, locus_tag_col: str = 'locus_tag', symbol_col: str = 'symbol',
                 synonyms_col: str = 'synonyms', sep: str = ',', synonyms_sep: str = '|') -> 'GeneNameMapper':
        """
        Builds a mapper from a CSV file with a locus tag, a symbol and a list of synonyms per line,
        such as the one written by :meth:`to_csv`.
        """
        mapper = cls()
        with _open_text(path) as handle:
            for row in csv.DictReader(handle, delimiter=sep):
                synonyms = (row.get(synonyms_col) or '').split(synonyms_sep)
                mapper.add(row[locus_tag_col], row.get(symbol_col), synonyms)
        return mapper

    def to_csv(self, path: PathLike, sep: str = ',', synonyms_sep: str = '|'):
        """
        Writes the mapper to a CSV file (locus_tag, symbol, synonyms) that :meth:`from_csv` reads,
        so that a mapper built from online sources can be reused offline.
        """
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, delimiter=sep)
            writer.writerow(['locus_tag', 'symbol', 'synonyms'])
            for tag in sorted(self._symbols):
                symbol = self._symbols[tag]
                synonyms = sorted(name for name in self._names[tag] if name not in (tag, symbol))
                writer.writerow([tag, symbol, synonyms_sep.join(synonyms)])

    # -----------------------------------------------------------------------------
    # Downloads
    # -----------------------------------------------------------------------------
    @staticmethod
    def download_ncbi_gene_info(organism: str, path: PathLike, division: str = 'Archaea_Bacteria') -> str:
        """
        Downloads an NCBI gene_info file.

        :param organism: the file name without extension, as listed at
        https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Archaea_Bacteria/
        (e.g. 'Escherichia_coli_str._K-12_substr._MG1655')
        :param path: where to save the .gz file
        :param division: the NCBI directory (e.g. 'Archaea_Bacteria', 'Fungi')
        :return: the path of the downloaded file
        """
        url = NCBI_GENE_INFO_URL.format(division=division, organism=organism)
        urllib.request.urlretrieve(url, path)
        return str(path)

    @staticmethod
    def download_kegg_list(organism: str, path: PathLike) -> str:
        """
        Downloads the KEGG gene list of an organism (e.g. 'eco' for E. coli K-12 MG1655).

        :return: the path of the downloaded file
        """
        urllib.request.urlretrieve(KEGG_LIST_URL.format(organism=organism), path)
        return str(path)
