"""
Engine for transcriptional regulatory networks reconstructed by text mining.

Text-mining pipelines, such as the @Note-based pipeline of R. T. Pereira (PhD thesis, University
of Minho, 2016), output regulatory events as edges: a regulator, a target and the trigger verb
found in the sentence, e.g.

    2,ArcA,sodA,be negatively regulate
    5,Crp,YggB,activate

This engine turns such edges into one Boolean rule per target and builds a RegulatoryModel:

- names are resolved into model identifiers (e.g. E. coli b-numbers) by a GeneNameMapper;
- the verb gives the sign: repression verbs (repress, inhibit, negatively regulate, ...) or
  activation verbs (activate, induce, ...); verbs with no clear sign ("regulate", "bind") are
  treated as activation by default, as in the original pipeline;
- activators are joined by OR and repressors negated:
  target = (A1 OR A2) AND NOT (R1 OR R2)

Edges that cannot be used (unresolved or ambiguous names, self-loops, ties between activation
and repression) are skipped and listed in the engine's `report`.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

from .boolean_csv import BooleanRegulatoryCSV
from .engines_utils import csv_warning

if TYPE_CHECKING:
    from mewpy.util.gene_names import GeneNameMapper

ACTIVATION = 'activation'
REPRESSION = 'repression'
UNKNOWN = 'unknown'

# Word stems matched inside the verb, checked in this order:
# relief of repression ('derepress', 'unblocked') before repression, and repression before
# activation, so that 'inactivate' is not read as 'activate'.
RELIEF_STEMS = ('derepress', 'unblock')
REPRESSION_STEMS = ('repress', 'inhibit', 'block', 'decreas', 'down-regulat', 'downregulat',
                    'down regulat', 'inactivat', 'suppress', 'negative', 'reduc', 'silenc', 'attenuat')
ACTIVATION_STEMS = ('activat', 'induc', 'increas', 'enhanc', 'stimulat', 'promot', 'up-regulat',
                    'upregulat', 'up regulat', 'positive', 'overexpress')


def verb_sign(verb: str,
              activation_stems: Sequence[str] = ACTIVATION_STEMS,
              repression_stems: Sequence[str] = REPRESSION_STEMS) -> str:
    """
    The sign of a regulatory event from its trigger verb: 'activation', 'repression' or 'unknown'.
    """
    verb = str(verb).lower()
    if any(stem in verb for stem in RELIEF_STEMS):
        return ACTIVATION
    if any(stem in verb for stem in repression_stems):
        return REPRESSION
    if any(stem in verb for stem in activation_stems):
        return ACTIVATION
    return UNKNOWN


@dataclass
class TextMiningReport:
    """What happened to the edges while building the rules."""
    edges: int = 0
    used: int = 0
    unresolved: Dict[str, Dict[str, int]] = field(default_factory=dict)
    self_loops: List[Tuple[str, str, str]] = field(default_factory=list)
    unknown_sign: List[Tuple[str, str, str]] = field(default_factory=list)
    conflicts: List[Tuple[str, str, Dict[str, int]]] = field(default_factory=list)

    def __str__(self):
        return (f'{self.edges} edges, {self.used} used; '
                f'{len(self.unresolved)} names unresolved, {len(self.self_loops)} self-loops, '
                f'{len(self.unknown_sign)} verbs without sign, {len(self.conflicts)} sign conflicts')


def edges_to_rules(edges: Iterable[Tuple[str, str, str]],
                   mapper: Optional['GeneNameMapper'] = None,
                   unknown_sign: str = ACTIVATION,
                   self_loops: bool = False,
                   activation_stems: Sequence[str] = ACTIVATION_STEMS,
                   repression_stems: Sequence[str] = REPRESSION_STEMS
                   ) -> Tuple[Dict[str, str], Dict[str, set], TextMiningReport]:
    """
    Converts (regulator, target, verb) edges into Boolean rules.

    :param edges: (regulator name, target name, verb) tuples
    :param mapper: resolves names into model identifiers; without it names are used as given
    :param unknown_sign: sign of verbs that are neither activation nor repression:
        'activation' (default, as in the original pipeline) or 'ignore' to skip those edges
    :param self_loops: keep edges whose regulator and target resolve to the same identifier.
        Off by default: they often come from sentences such as "yiiP encodes FieF"
    :param activation_stems: word stems that mark an activation verb
    :param repression_stems: word stems that mark a repression verb
    :return: rules (target id -> rule), names (id -> names found in the text) and a report
    """
    if unknown_sign not in (ACTIVATION, 'ignore'):
        raise ValueError(f"unknown_sign must be 'activation' or 'ignore', not {unknown_sign!r}")

    report = TextMiningReport()
    names = defaultdict(set)
    votes = defaultdict(Counter)

    def resolve(name):
        name = str(name).strip()
        if mapper is None:
            return name.replace(' ', '')
        identifier = mapper.resolve(name)
        if identifier is None:
            report.unresolved[name] = mapper.candidates(name)
        return identifier

    for regulator_name, target_name, verb in edges:
        report.edges += 1
        regulator, target = resolve(regulator_name), resolve(target_name)
        if regulator is None or target is None:
            continue

        names[regulator].add(str(regulator_name).strip())
        names[target].add(str(target_name).strip())

        if regulator == target and not self_loops:
            report.self_loops.append((regulator_name, target_name, verb))
            continue

        sign = verb_sign(verb, activation_stems, repression_stems)
        if sign == UNKNOWN:
            report.unknown_sign.append((regulator_name, target_name, verb))
            if unknown_sign == 'ignore':
                continue
            sign = ACTIVATION

        votes[(regulator, target)][sign] += 1
        report.used += 1

    activators, repressors = defaultdict(list), defaultdict(list)
    for (regulator, target), counter in votes.items():
        if counter[ACTIVATION] > counter[REPRESSION]:
            activators[target].append(regulator)
        elif counter[REPRESSION] > counter[ACTIVATION]:
            repressors[target].append(regulator)
        else:
            report.conflicts.append((regulator, target, dict(counter)))

    rules = {target: boolean_rule(activators[target], repressors[target])
             for target in sorted(set(activators) | set(repressors))}

    return rules, dict(names), report


def boolean_rule(activators: Iterable[str], repressors: Iterable[str]) -> str:
    """
    The rule of a target: any activator switches it on, any repressor switches it off.

        boolean_rule(['A1', 'A2'], ['R1'])  ->  '(A1 OR A2) AND NOT R1'
        boolean_rule([], ['R1', 'R2'])      ->  'NOT (R1 OR R2)'
    """
    activators, repressors = sorted(set(activators)), sorted(set(repressors))

    def either(ids):
        return ids[0] if len(ids) == 1 else '(' + ' OR '.join(ids) + ')'

    if activators and repressors:
        return f'{either(activators)} AND NOT {either(repressors)}'
    if repressors:
        return f'NOT {either(repressors)}'
    return ' OR '.join(activators)


class TextMiningRegulatoryCSV(BooleanRegulatoryCSV):

    def __init__(self, io, config, model=None):
        """
        Engine for regulatory edges obtained by text mining (regulator, target, verb).

        Reader arguments: sep (default ','), header (default None: no header row).

        Passed in the Reader's `config` dictionary (all optional):
            edge_regulator_col, edge_target_col, edge_verb_col: column positions (default 1, 2, 3,
                the layout 'id, regulator, target, verb' of the @Note pipeline)
            mapper: a mewpy.util.gene_names.GeneNameMapper to resolve names into model identifiers
            unknown_sign: 'activation' (default) or 'ignore'
            self_loops: keep self-regulation edges (default False)

        Example:
            Reader(Engines.TextMiningRegulatoryCSV, 'edges.csv', config={'mapper': mapper})

        After reading, `reader.engine.report` lists the edges that were skipped and why.
        """
        super().__init__(io, config, model)
        self.report = None

    def build_data_frame(self):
        # the Reader always sets target_col/regulator_col for other engines, hence the edge_ prefix
        sep = self.config.get('sep', ',')
        header = self.config.get('header', None)
        regulator_col = self.config.get('edge_regulator_col', 1)
        target_col = self.config.get('edge_target_col', 2)
        verb_col = self.config.get('edge_verb_col', 3)

        try:
            df = pd.read_csv(self.io, sep=sep, header=header, dtype=str, skipinitialspace=True)
        except BaseException as exc:
            self.clean()
            self.close()
            raise exc

        df = df.fillna('')
        edges = zip(df.iloc[:, regulator_col], df.iloc[:, target_col], df.iloc[:, verb_col])

        rules, names, report = edges_to_rules(edges,
                                              mapper=self.config.get('mapper'),
                                              unknown_sign=self.config.get('unknown_sign', ACTIVATION),
                                              self_loops=self.config.get('self_loops', False))
        self.report = report

        if report.unresolved:
            self.warnings.append(partial(csv_warning,
                                         f'{len(report.unresolved)} names could not be resolved: '
                                         f'{", ".join(sorted(report.unresolved)[:10])}'))
        if report.conflicts:
            self.warnings.append(partial(csv_warning,
                                         f'{len(report.conflicts)} regulator-target pairs skipped: '
                                         f'as many activation as repression mentions'))

        # one alias column per name found in the text (e.g. 'FtnA' and 'ftnA' for b1905)
        targets = list(rules)
        aliases = [sorted(names.get(target, ())) for target in targets]
        width = max((len(a) for a in aliases), default=0)
        columns = {'ids': targets, 'rules': [rules[t] for t in targets]}
        for i in range(width):
            columns[f'aliases_{i + 1}'] = [a[i] if i < len(a) else '' for a in aliases]

        self.dto.data_frame = pd.DataFrame(columns, index=targets)
        self.dto.aliases_columns = [f'aliases_{i + 1}' for i in range(width)]
