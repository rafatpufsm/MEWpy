import gzip
import os
import tempfile
import unittest

from mewpy.util.gene_names import GeneNameMapper, normalize_name

# NCBI gene_info format (trailing columns dropped). arcA, fnr and pdhR follow the real E. coli K-12
# MG1655 file; genX, gpp/spoT and noTag are simplified rows that reproduce an ambiguous synonym,
# a synonym that is the official symbol of another gene, and a gene without a locus tag.
NCBI_GENE_INFO = (
    '#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tmap_location\tdescription\t'
    'type_of_gene\tSymbol_from_nomenclature_authority\n'
    '511145\t948874\tarcA\tb4401\tECK4393|cpxC|dye|fexA|msp|seg|sfrA\t-\t-\t-\t-\tprotein-coding\tarcA\n'
    '511145\t945908\tfnr\tb1334\tECK1330|nirA|nirR|oxrA\t-\t-\t-\t-\tprotein-coding\tfnr\n'
    '511145\t944834\tpdhR\tb0113\tECK0112|aceC|genA|yacB\t-\t-\t-\t-\tprotein-coding\tpdhR\n'
    '511145\t900001\tgenX\tb9001\tseg\t-\t-\t-\t-\tprotein-coding\tgenX\n'
    '511145\t900002\tgpp\tb3779\tgppA\t-\t-\t-\t-\tprotein-coding\tgpp\n'
    '511145\t900003\tspoT\tb3650\tgpp\t-\t-\t-\t-\tprotein-coding\tspoT\n'
    '511145\t900004\tnoTag\t-\t-\t-\t-\t-\t-\tprotein-coding\tnoTag\n'
)

KEGG_LIST = (
    'eco:b4401\tCDS\tcomplement(4639590..4640306)\tarcA, cpxC, dye; DNA-binding transcriptional dual regulator ArcA\n'
    'eco:b0114\tCDS\t123017..125680\taceE; pyruvate dehydrogenase E1 component\n'
    'eco:b9999\tCDS\t1..10\tuncharacterized protein\n'
)


class TestGeneNameMapper(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gene_info = os.path.join(self.tmp.name, 'ecoli.gene_info.gz')
        with gzip.open(self.gene_info, 'wt', encoding='utf-8') as handle:
            handle.write(NCBI_GENE_INFO)
        self.kegg = os.path.join(self.tmp.name, 'eco.tsv')
        with open(self.kegg, 'w', encoding='utf-8') as handle:
            handle.write(KEGG_LIST)
        self.mapper = GeneNameMapper.from_ncbi_gene_info(self.gene_info)

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalize(self):
        self.assertEqual(normalize_name(' ArcA '), 'arca')
        self.assertEqual(normalize_name('"Fnr"'), 'fnr')

    def test_symbol_protein_name_and_locus_tag(self):
        self.assertEqual(self.mapper.resolve('arcA'), 'b4401')
        # protein names differ from the gene symbol only in case
        self.assertEqual(self.mapper.resolve('ArcA'), 'b4401')
        self.assertEqual(self.mapper.resolve('Fnr'), 'b1334')
        # a locus tag resolves to itself
        self.assertEqual(self.mapper.resolve('b0113'), 'b0113')
        self.assertEqual(self.mapper.symbol('b0113'), 'pdhR')

    def test_synonyms(self):
        self.assertEqual(self.mapper.resolve('sfrA'), 'b4401')
        self.assertEqual(self.mapper.resolve('PdhR'), 'b0113')
        self.assertEqual(self.mapper.resolve('aceC'), 'b0113')

    def test_ambiguous_synonym_is_not_resolved(self):
        # 'seg' is a synonym of arcA and of genX
        self.assertIsNone(self.mapper.resolve('seg'))
        self.assertEqual(self.mapper.candidates('seg'), {'b4401': 2, 'b9001': 2})

    def test_symbol_wins_over_synonym_of_another_gene(self):
        # 'gpp' is the symbol of b3779 and a synonym of spoT (b3650)
        self.assertEqual(self.mapper.resolve('gpp'), 'b3779')

    def test_unknown_and_missing_locus_tag(self):
        self.assertIsNone(self.mapper.resolve('lacZ'))
        self.assertIsNone(self.mapper.resolve('noTag'))
        self.assertEqual(len(self.mapper), 6)

    def test_resolve_many(self):
        resolved, unresolved = self.mapper.resolve_many(['ArcA', 'seg', 'lacZ'])
        self.assertEqual(resolved, {'ArcA': 'b4401'})
        self.assertEqual(set(unresolved), {'seg', 'lacZ'})
        self.assertEqual(unresolved['lacZ'], {})

    def test_kegg_list_and_merge(self):
        kegg = GeneNameMapper.from_kegg_list(self.kegg)
        self.assertEqual(kegg.resolve('AceE'), 'b0114')
        self.assertEqual(kegg.resolve('cpxC'), 'b4401')
        # a gene without a name is still known by its locus tag
        self.assertEqual(kegg.resolve('b9999'), 'b9999')

        self.mapper.update(kegg)
        self.assertEqual(self.mapper.resolve('aceE'), 'b0114')
        self.assertEqual(self.mapper.resolve('sfrA'), 'b4401')

    def test_csv_round_trip(self):
        path = os.path.join(self.tmp.name, 'names.csv')
        self.mapper.to_csv(path)
        loaded = GeneNameMapper.from_csv(path)

        self.assertEqual(loaded.locus_tags, self.mapper.locus_tags)
        for name in ['ArcA', 'sfrA', 'dye', 'PdhR', 'gpp', 'b1334']:
            self.assertEqual(loaded.resolve(name), self.mapper.resolve(name), name)
        self.assertIsNone(loaded.resolve('seg'))


if __name__ == '__main__':
    unittest.main()
