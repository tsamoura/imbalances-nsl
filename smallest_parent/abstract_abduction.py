from pysdd.sdd import SddNode
from .formula import createConjunction, createDisjunction
from .abstract_translator import AbstractTranslator


class AbstractAbduction(object):
    def __init__(self, translator: AbstractTranslator):
        self.translator = translator
        self.prepareTheory = None
        self.cache_sdds = dict()
        self.cache_proofs = dict()
        self.scenario = 0

    def abduce(self, target) -> SddNode:
        if not (target in self.cache_sdds):
            self.cache_sdds[target] = self.callSolver(target)
        return self.cache_sdds[target]

    def callSolver(self, target) -> SddNode:
        proofs = self.solve(target)

        index = 0
        disjuncts = list()
        while index < len(proofs):
            proof = proofs[index]
            conjunction = self.convertProofToAbductiveFormula(proof)
            disjuncts.append(conjunction)
            index += 1

        return createDisjunction(disjuncts)

    def convertProofToAbductiveFormula(self, proof) -> SddNode:
        literals = list()
        for abducible in proof:
            literal = self.translator.getSddLiteral(abducible)
            literals.append(literal)
            for negated in self.translator.getMutuallyExclusiveAbducibles(abducible):
                literals.append(self.translator.getSddLiteral(negated).negate())
        return createConjunction(literals)

    def solve(self, target, output_indices = False):
        if not (target in self.cache_proofs):
            self.cache_proofs[target] = self._solve(target, output_indices)
        return self.cache_proofs[target]

    def _solve(self, target):
        pass
