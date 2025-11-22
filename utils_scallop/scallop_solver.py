import scallopy
import torch.nn as nn
from typing import Tuple


class ScallopSolver(nn.Module):
    def __init__(self, network, index_to_network_class, index_to_output_class, args):
        super(ScallopSolver, self).__init__()
        self.network = network
        self.index_to_network_class = index_to_network_class
        self.index_to_output_class = index_to_output_class
        self.args = args
        self.M = args.M
        self.scl_ctx = scallopy.ScallopContext(
            provenance="difftopkproofsdebug", k=args.top_k
        )
        self.input_mapping = [c for _, c in self.index_to_network_class.items()]
        self.output_mapping = [(c,) for _, c in self.index_to_output_class.items()]
        self.forward_function = None

    def set_context_forward_function(self, predicate: str):
        self.forward_function = self.scl_ctx.forward_function(
            predicate,
            output_mapping=self.output_mapping,  # [(i,) for i in range(domain)],
            jit=self.args.jit,
            dispatch=self.args.dispatch,
        )

    # def forward(self, x: Tuple):
    #    digits = {f"digit_{i+1}": self.network(x[i]) for i in range(self.M)}
    #    arguments = {**digits}
    #    return self.forward_function(**arguments), []  # Tensor 64 x 19

    def forward(self, x: Tuple):
        batch_size = x[0].shape[0]
        predictions = [self.network(x[i]) for i in range(self.M)]
        arguments, mappings = self.prepare_context_inputs(
            predictions,
            batch_size,
        )
        output_probabilities, proofs = self.forward_function(**arguments)
        transformed_proofs = self.convert_proofs_to_internal_format(
            proofs, mappings, batch_size
        )
        return predictions, output_probabilities, transformed_proofs

    def prepare_context_inputs(self, predictions, batch_size):
        inputs = {c: None for _, c in self.index_to_input_predicates.items()}
        mappings = {k: None for k in range(self.M)}
        for k in range(self.M):
            inputs[self.index_to_input_predicates[k]] = [
                [
                    ((predictions[k][b][i], 1 + i + k * len(self.input_mapping)), (c,))
                    for i, c in self.index_to_network_class.items()
                ]
                for b in range(batch_size)
            ]
            mappings[k] = [
                {
                    1 + i + k * len(self.input_mapping): c
                    for i, c in self.index_to_network_class.items()
                }
                for b in range(batch_size)
            ]
        return {**inputs}, mappings

    def convert_proofs_to_internal_format(self, proofs, mappings, batch_size):
        transformed = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            transformed[b] = [[] for _ in range(len(self.index_to_output_class))]
            for d, _ in self.index_to_output_class.items():
                transformed[b][d] = [
                    [
                        f"at({mappings[k][b][proofs[b][d][p][k][1]]},{k})"
                        for k in range(self.M)
                    ]
                    for p in range(len(proofs[b][d]))
                ]
        return transformed
