import os
import tempfile
import unittest
from pathlib import Path

from mewpy.io.engines.text_mining_csv import (ACTIVATION, REPRESSION, UNKNOWN, boolean_rule,
                                              edges_to_rules, verb_sign)
from mewpy.util.gene_names import GeneNameMapper

MODELS_PATH = Path(__file__).parent.joinpath('data')
EC_CORE_MODEL = MODELS_PATH.joinpath('e_coli_core.xml')

# Edges extracted from PubMed abstracts by the @Note pipeline (Listing 4.3 of R. T. Pereira's
# PhD thesis, University of Minho, 2016): id, regulator, target, verb
THESIS_EDGES = """3, yiiP, FieF, encode
2, ArcA, sodA, be negatively regulate
10, OxyR, Fur, activate
1, Fur, fhuF, regulate
7, Fur, YggB, activate
6, Fur, Cfa, activate
5, Crp, YggB, activate
4, Crp, Cfa, activate
9, SoxR, ArcA, regulate
8, Fur, fhuA, be express
13, IscR, Fur, to bind
11, Fur, FtnA, be induce
12, Fur, ftnA, be induce
"""


def thesis_mapper():
    """Names of the thesis edges and their E. coli b-numbers (yiiP and FieF are one gene)."""
    mapper = GeneNameMapper()
    for tag, symbol, synonyms in [('b3915', 'fieF', ['yiiP']), ('b4401', 'arcA', []),
                                  ('b3908', 'sodA', []), ('b3961', 'oxyR', []),
                                  ('b0683', 'fur', []), ('b4367', 'fhuF', []),
                                  ('b2924', 'mscS', ['yggB']), ('b1661', 'cfa', []),
                                  ('b3357', 'crp', []), ('b4063', 'soxR', []),
                                  ('b0150', 'fhuA', []), ('b2531', 'iscR', []),
                                  ('b1905', 'ftnA', [])]:
        mapper.add(tag, symbol, synonyms)
    return mapper


class TestTextMiningRules(unittest.TestCase):

    def test_verb_sign(self):
        self.assertEqual(verb_sign('activate'), ACTIVATION)
        self.assertEqual(verb_sign('be induce'), ACTIVATION)
        self.assertEqual(verb_sign('be negatively regulate'), REPRESSION)
        self.assertEqual(verb_sign('inhibits'), REPRESSION)
        # repression is checked before activation
        self.assertEqual(verb_sign('inactivates'), REPRESSION)
        # relief of repression is an activation
        self.assertEqual(verb_sign('derepresses'), ACTIVATION)
        self.assertEqual(verb_sign('unblocked'), ACTIVATION)
        self.assertEqual(verb_sign('regulate'), UNKNOWN)
        self.assertEqual(verb_sign('to bind'), UNKNOWN)

    def test_boolean_rule(self):
        self.assertEqual(boolean_rule(['A'], []), 'A')
        self.assertEqual(boolean_rule(['B', 'A'], []), 'A OR B')
        self.assertEqual(boolean_rule([], ['R']), 'NOT R')
        self.assertEqual(boolean_rule([], ['R2', 'R1']), 'NOT (R1 OR R2)')
        self.assertEqual(boolean_rule(['A1', 'A2'], ['R']), '(A1 OR A2) AND NOT R')

    def test_thesis_edges_give_thesis_rules(self):
        edges = [line.split(', ')[1:] for line in THESIS_EDGES.strip().splitlines()]
        rules, names, report = edges_to_rules(edges, mapper=thesis_mapper())

        # the Boolean rules of Listing 4.4 of the thesis, written with b-numbers
        self.assertEqual(rules, {'b0150': 'b0683',                # fhuA = Fur
                                 'b0683': 'b2531 OR b3961',       # Fur = OxyR OR IscR
                                 'b1661': 'b0683 OR b3357',       # Cfa = Fur OR Crp
                                 'b1905': 'b0683',                # ftnA = FtnA = Fur
                                 'b2924': 'b0683 OR b3357',       # YggB = Fur OR Crp
                                 'b3908': 'NOT b4401',            # sodA = NOT ArcA
                                 'b4367': 'b0683',                # fhuF = Fur
                                 'b4401': 'b4063'})               # ArcA = SoxR
        # 'yiiP encodes FieF' names one gene twice: not a regulation
        self.assertEqual(report.self_loops, [('yiiP', 'FieF', 'encode')])
        self.assertEqual(report.edges, 13)
        self.assertEqual(report.used, 12)
        self.assertEqual(len(report.unknown_sign), 4)
        self.assertEqual(names['b1905'], {'FtnA', 'ftnA'})

    def test_unknown_sign_ignore(self):
        edges = [('SoxR', 'ArcA', 'regulate'), ('OxyR', 'Fur', 'activate')]
        rules, _, report = edges_to_rules(edges, mapper=thesis_mapper(), unknown_sign='ignore')
        self.assertEqual(rules, {'b0683': 'b3961'})
        self.assertEqual(report.used, 1)

    def test_majority_and_conflicts(self):
        edges = [('Fur', 'fhuF', 'represses'), ('Fur', 'fhuF', 'inhibits'), ('Fur', 'fhuF', 'activates'),
                 ('Crp', 'cfa', 'activates'), ('Crp', 'cfa', 'represses')]
        rules, _, report = edges_to_rules(edges, mapper=thesis_mapper())
        self.assertEqual(rules, {'b4367': 'NOT b0683'})
        self.assertEqual(report.conflicts, [('b3357', 'b1661', {ACTIVATION: 1, REPRESSION: 1})])

    def test_unresolved_names_are_skipped(self):
        edges = [('LacI', 'lacZ', 'represses'), ('Fur', 'fhuF', 'regulates')]
        rules, _, report = edges_to_rules(edges, mapper=thesis_mapper())
        self.assertEqual(rules, {'b4367': 'b0683'})
        self.assertEqual(set(report.unresolved), {'LacI', 'lacZ'})

    def test_without_mapper_names_are_identifiers(self):
        rules, _, _ = edges_to_rules([('ArcA', 'sodA', 'represses')])
        self.assertEqual(rules, {'sodA': 'NOT ArcA'})


class TestTextMiningEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.edges = os.path.join(self.tmp.name, 'edges.csv')
        with open(self.edges, 'w', encoding='utf-8') as handle:
            handle.write(THESIS_EDGES)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_regulatory_model(self):
        from mewpy.io import Reader, Engines, read_model

        reader = Reader(Engines.TextMiningRegulatoryCSV, self.edges, config={'mapper': thesis_mapper()})
        model = read_model(reader)

        self.assertEqual(len(model.interactions), 8)
        self.assertIn('b4401', model.regulators)
        sod_a = model.interactions['b3908_interaction']
        self.assertEqual(set(sod_a.regulators), {'b4401'})
        self.assertIn('sodA', model.targets['b3908'].aliases)
        self.assertEqual(reader.engine.report.used, 12)

    def test_custom_columns_and_header(self):
        from mewpy.io import Reader, Engines, read_model

        path = os.path.join(self.tmp.name, 'edges.tsv')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('target\tverb\tregulator\nsodA\trepresses\tArcA\n')

        reader = Reader(Engines.TextMiningRegulatoryCSV, path, sep='\t', header=0,
                        config={'edge_target_col': 0, 'edge_verb_col': 1, 'edge_regulator_col': 2})
        model = read_model(reader)
        self.assertEqual(model.interactions['sodA_interaction'].regulatory_events[1].to_string(), '( ~ ArcA)')

    def test_integrated_model(self):
        from mewpy.io import Reader, Engines, read_model

        model = read_model(Reader(Engines.MetabolicSBML, EC_CORE_MODEL),
                           Reader(Engines.TextMiningRegulatoryCSV, self.edges,
                                  config={'mapper': thesis_mapper()}))
        self.assertTrue(model.is_metabolic() and model.is_regulatory())
        self.assertEqual(len(model.reactions), 95)
        self.assertEqual(len(model.interactions), 8)


if __name__ == '__main__':
    unittest.main()
