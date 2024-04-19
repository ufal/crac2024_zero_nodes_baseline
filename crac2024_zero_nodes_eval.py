#!/usr/bin/env python3

# This file is part of CRAC24 Zero Nodes Baseline
# <https://github.com/ufal/crac2024_zero_nodes_baseline>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import collections
from dataclasses import dataclass
import re

@dataclass
class F1Score:
    f1: float
    p: float
    r: float

    def __init__(self, tp: int, fp: int, fn: int):
        self.p = tp / (tp + fp) if tp + fp > 0 else 0
        self.r = tp / (tp + fn) if tp + fn > 0 else 0
        self.f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0


def evaluate(system_conllu: str, gold_conllu: str) -> dict[str, F1Score]:
    @dataclass
    class ZeroNode:
        sentence: int
        word_order: int
        head: int
        deprel: str

    def load_conllu(conllu: str) -> list[ZeroNode]:
        sentence = 0
        zero_nodes = []
        for line in re.split(r"\r*\n", conllu):
            if not line:
                sentence += 1
                continue
            if not re.match(r"^[0-9]*[.]", line):
                continue
            columns = line.split("\t")
            word_order = columns[0].split(".", maxsplit=1)[0]
            head, deprel = columns[8].split("|", maxsplit=1)[0].split(":", maxsplit=1)
            zero_nodes.append(ZeroNode(sentence, int(word_order), int(head), deprel))
        return zero_nodes

    system = load_conllu(system_conllu)
    gold = load_conllu(gold_conllu)

    metrics = {}
    for metric, get_example in [
            ("WO", lambda zero_node: (zero_node.sentence, zero_node.word_order)),
            ("ARC", lambda zero_node: (zero_node.sentence, zero_node.head)),
            ("DEP", lambda zero_node: (zero_node.sentence, zero_node.head, zero_node.deprel)),
            ("WO_DEP", lambda zero_node: (zero_node.sentence, zero_node.word_order, zero_node.head, zero_node.deprel)),
    ]:
        system_examples = collections.Counter(get_example(zero_node) for zero_node in system)
        gold_examples = collections.Counter(get_example(zero_node) for zero_node in gold)
        metrics[metric] = F1Score(
            tp=(system_examples & gold_examples).total(),
            fp=(system_examples - gold_examples).total(),
            fn=(gold_examples - system_examples).total(),
        )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, help="Path to the system output CoNLL-U file.")
    parser.add_argument("gold", type=str, help="Path to the gold CoNLL-U file.")
    args = parser.parse_args()

    with open(args.system, "r", encoding="utf-8-sig") as system_file:
        system_conllu = system_file.read()
    with open(args.gold, "r", encoding="utf-8-sig") as gold_file:
        gold_conllu = gold_file.read()
    metrics = evaluate(system_conllu, gold_conllu)
    for metric, score in metrics.items():
        print("{}: f1={:.2f}%, p={:.2f}%, r={:.2f}%".format(metric, 100 * score.f1, 100 * score.p, 100 * score.r))
