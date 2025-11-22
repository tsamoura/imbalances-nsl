from .scallop_solver import ScallopSolver
from .arithmetic_solver import ArithmeticSolver
from typing import Tuple
import torch
import torch.nn.functional as F


class HWFSolver(ArithmeticSolver):
    def __init__(
        self, network, index_to_network_class, index_to_output_class, args, m=0.9
    ):
        ScallopSolver.__init__(
            self, network, index_to_network_class, index_to_output_class, args
        )

        if not args.M in [3,5,7]:
            raise ValueError("Illegal number of input symbols. Only 3,5,7 are supported.")

        self.index_to_input_predicates = {k: f"symbol{k+1}" for k in range(self.M)}

        for i in range(self.M):
            self.scl_ctx.add_relation(
                f"symbol{i+1}", int, self.input_mapping  # input_mapping=list(range(10))
            )

        if args.M == 3:
            self.scl_ctx.add_rule("arithmetic(x1+x2) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, x1 < 10, x2 < 10") 
            self.scl_ctx.add_rule("arithmetic(x1-x2) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, x1 < 10, x2 < 10") 
            self.scl_ctx.add_rule("arithmetic(x1*x2) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, x1 < 10, x2 < 10")

        if args.M == 5:
            self.scl_ctx.add_rule("arithmetic((x1+x2)+x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 10, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1-x2)+x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 10, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1*x2)+x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 10, x1 < 10, x2 < 10, x3 < 10") 
            
            self.scl_ctx.add_rule("arithmetic((x1+x2)-x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 11, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1-x2)-x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 11, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1*x2)-x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 11, x1 < 10, x2 < 10, x3 < 10") 

            self.scl_ctx.add_rule("arithmetic((x1+x2)*x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 12, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1-x2)*x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 12, x1 < 10, x2 < 10, x3 < 10") 
            self.scl_ctx.add_rule("arithmetic((x1*x2)*x3) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 12, x1 < 10, x2 < 10, x3 < 10") 

        if args.M == 7:
            self.scl_ctx.add_rule("arithmetic(((x1+x2)+x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)+x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)+x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1+x2)-x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)-x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)-x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1+x2)*x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)*x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)*x3)+x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 10, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 

            self.scl_ctx.add_rule("arithmetic(((x1+x2)+x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)+x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)+x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1+x2)-x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)-x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)-x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1+x2)*x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)*x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)*x3)-x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 11, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 

            self.scl_ctx.add_rule("arithmetic(((x1+x2)+x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)+x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)+x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 10, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10")             
            self.scl_ctx.add_rule("arithmetic(((x1+x2)-x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)-x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)-x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 11, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1+x2)*x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 10, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1-x2)*x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 11, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 
            self.scl_ctx.add_rule("arithmetic(((x1*x2)*x3)*x4) :- symbol1(x1), symbol2(op1), symbol3(x2), op1 == 12, symbol4(op2), symbol5(x3), op2 == 12, symbol6(op3), symbol7(x4), op3 == 12, x1 < 10, x2 < 10, x3 < 10, x4 < 10") 

        self.set_context_forward_function("arithmetic")

        # Records parameters
        self.feat_mean = None
        self.m = m

    # TODO to support RECORDS for other benchamrk, the neural networks must have a common forward interface as MNIST.
    def forward(self, x: Tuple):
        batch_size = x[0].shape[0]
        if self.args.records:
            logits_x = [
                self.network(x[i], do_softmax=False, eval_only=False)
                for i in range(self.M)
            ]
            feats_x = torch.stack([feat for _, feat in logits_x])
            current_mean = torch.sum(feats_x, dim=0) / self.args.M
            if self.feat_mean is None:
                self.feat_mean = (1 - self.m) * (current_mean).detach().mean(0)
            else:
                self.feat_mean = self.m * self.feat_mean + (1 - self.m) * (
                    current_mean
                ).detach().mean(0)

            bias = self.network.fc(self.feat_mean.view(-1, 16 * 4 * 4)).detach()
            bias = F.softmax(bias, dim=1)
            predictions = [logits - torch.log(bias + 1e-9) for logits, _ in logits_x]
            predictions = [F.softmax(logits, dim=1) for logits, _ in logits_x]
        else:
            logits_x = [
                self.network(x[i], do_softmax=False, eval_only=True)
                for i in range(self.M)
            ]
            predictions = [F.softmax(logits, dim=1) for logits in logits_x]

        arguments, mappings = self.prepare_context_inputs(
            predictions,
            batch_size,
        )
        output_probabilities, proofs = self.forward_function(**arguments)
        transformed_proofs = self.convert_proofs_to_internal_format(
            proofs, mappings, batch_size
        )
        return logits_x, predictions, output_probabilities, transformed_proofs
