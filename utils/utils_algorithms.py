import torch
import math
from ortools.linear_solver import pywraplp
import sys
sys.path.append("../")

# The following provide the linear inequalities corresponding to the abductive proofs
# Let \phi_1, \dots, \phi_n be the number of proofs
# Each proof is a conjunction of facts, \phi_l = \f_{l,1}, \dots, \f_{l,n_l}
# The Tseitin transformation results in the following DNF formula
# \bigwedge_{l=1}^n \alpha_l (*)
# \wedge
# \bigwedge_{l=1}^n \alpha_l <-> \f_{l,1}, \dots, \f_{l,n_l}
# The second conjunction results in the following clauses:
# (Type I)
# \alpha_l -> \f_{l,1}, \dots, \f_{l,n_l} = \neg \alpha_l \vee (\f_{l,1}, \dots, \f_{l,n_l}) =
# (\neg \alpha_l \vee \f_{l,1} ) \wedge \dots \wedge (\neg \alpha_l \vee \f_{l,n_l} ) (**)
# (Type II)
# \f_{l,1}, \dots, \f_{l,n_l} -> \alpha_l =
# \neg \f_{l,1} \vee \dots \vee \neg \f_{l,n} \vee \alpha_l (***)

# Based on the above, we have the following linear equations:
# Inequality I: a_1 + \dots a_n \geq 1 <= computed out of (*)
# For each clause in (**), we have the following linear inequality:
# Inequality II: 1- \alpha_l + \f_{l,j} \geq 1
# For (***), we have the following linear inequality:
# Inequality III: 1 - \f_{l,1} + \dots + 1 - \f_{l,n} + \alpha_l >= 1
# We additionally, have the following linear inequalities to ensure that only one label each assigned to one
# for each input object x_i
# Inequality IV: q_{i,1} + \dots + q_{i,m} = 1, where q_{i,j} denotes that x_i is assigned to label j.

# In addition, we need to add linear inequaldities for the objective, and the distribution constraints,
# where each distribution constraint makes sure that the distribution of a class should be as in the emp_dist.



# proofs is a list of the candidate proofs corresponding to each input training sample
# Implements the relaxation via https://developers.google.com/optimization/mip/mip_example#comparing_linear_and_integer_optimization
# If emp_dist == None, then we do not introduce distribution constraints to the lp.
def ilp_pywraplp_mipll(
    proofs_n,
    index_to_network_class,
    predictions_n,
    emp_dist=None,
    epsilon_ilp=0.99,
    continuous_relaxation=False,
    domain_restrictions=None
):
    epsilon = epsilon_ilp

    if domain_restrictions is None:
        def one(c,j):
            return 1
        domain_restrictions = one    

    while True:
        # Create the mip solver with the SCIP backend.
        if not continuous_relaxation:
            solver = pywraplp.Solver.CreateSolver("SCIP")
        else:
            solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            return

        # n denotes the number of input training samples
        # m denotes the number of classes each input can be assigned to
        # M is the number of instances per input sample
        # We assume that all x_i's share classes of the same domain
        n = predictions_n[0].shape[0]
        m = predictions_n[0].shape[1]
        M = len(predictions_n)

        # Assert that the proofs that are passed to the linear solvers should correspond to those of the
        # elements in the queue
        for j in range(M):
            assert predictions_n[j].shape[0] == len(proofs_n)

        fact_to_var = list()
        for s in range(n):
            # "x_{j,k}" is the variable associated to the j-th input instance, k-th class
            # The j-th list provides the variables for j-th input
            if not continuous_relaxation:
                fact_to_var.append(
                    {
                        f"at({index_to_network_class[k]},{j})": solver.IntVar(
                            0, domain_restrictions(index_to_network_class[k],j), f"at{s}_{index_to_network_class[k]}_{j}"
                        )
                        for k in range(m)
                        for j in range(M)
                    }
                )
            else:
                fact_to_var.append(
                    {
                        f"at({index_to_network_class[k]},{j})": solver.NumVar(
                            0, domain_restrictions(index_to_network_class[k],j), f"at{s}_{index_to_network_class[k]}_{j}"
                        )
                        for k in range(m)
                        for j in range(M)
                    }
                )

        # Organize the variables in lists
        # This will help us compute the solution
        variables = list()
        for j in range(M):
            variables.append(
                [
                    [
                        fact_to_var[s][f"at({index_to_network_class[k]},{j})"]
                        for k in range(m)
                    ]
                    for s in range(n)
                ]
            )

        # "a_{s,l}" is the variable associated to the s-th training sample, l-th proof
        if not continuous_relaxation:
            alphas = [
                [solver.IntVar(0, 1, f"a{s}_{l}") for l in range(len(proofs_n[s]))]
                for s in range(n)
            ]
        else:
            alphas = [
                [solver.NumVar(0, 1, f"a{s}_{l}") for l in range(len(proofs_n[s]))]
                for s in range(n)
            ]

        # s in an index of training samples
        for s in range(n):
            v_s = fact_to_var[s]
            alpha_s = alphas[s]
            # i is an index of the proofs of the s-th training sample
            for l in range(len(proofs_n[s])):
                # The facts that are true in the l-th proof
                # Each fact is mapped to variables of the solver
                f_l = proofs_n[s][l]

                # For each fact add inequality II: 1- \alpha_l + \f_{l,j} \geq 1
                [solver.Add(1 - alpha_s[l] + v_s[f_l[j]] >= 1) for j in range(len(f_l))]

                # Add inequality III: 1 - \f_{l,1} + \dots + 1 - \f_{l,n} + \alpha_i >= 1
                solver.Add(
                    sum([1 - v_s[f_l[j]] for j in range(len(f_l))]) + alpha_s[l] >= 1
                )

            # For the s-th training sample, add inequality I: a_1 + \dots a_n \geq 1 that varies over all the proofs for the s-th sample
            solver.Add(sum([alpha_s[l] for l in range(len(proofs_n[s]))]) >= 1)

        # For each input object x_j add
        # Inequality IV: x_{j,1} + \dots + x_{j,m} = 1.
        # These are the classification constraints
        for s in range(n):
            v_s = fact_to_var[s]
            for j in range(M):
                solver.Add(
                    sum([v_s[f"at({index_to_network_class[k]},{j})"] for k in range(m)])
                    == 1
                )

        if emp_dist is not None:
            # Create the distribution constraints
            for j in range(M):
                for k in range(m):
                    constraint = [
                        fact_to_var[s][f"at({index_to_network_class[k]},{j})"]
                        for s in range(n)
                    ]
                    solver.Add(sum(constraint) >= emp_dist[k].item() * n - epsilon)
                    solver.Add(sum(constraint) <= emp_dist[k].item() * n + epsilon)

        # Create the objective expression
        objective_expr = []
        for j in range(M):
            score2 = predictions_n[j].tolist()
            for k in range(m):
                objective_expr.extend(
                    [
                        fact_to_var[s][f"at({index_to_network_class[k]},{j})"]
                        * math.log(score2[s][k])
                        for s in range(n)
                        if score2[s][k] > 0
                    ]
                )

        solver.Minimize(-solver.Sum(objective_expr))

        # Prints the program in the console
        # print(solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','), sep='\n')

        # print(f"Solving with {solver.SolverVersion()}")
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            # print("Solution:")
            # print("Objective value =", solver.Objective().Value())
            # print(f"Problem solved in {solver.wall_time():d} milliseconds")
            Q = [
                [
                    [variables[j][s][k].solution_value() for k in range(m)]
                    for s in range(n)
                ]
                for j in range(M)
            ]

            return [torch.Tensor(Q[j]).cuda() for j in range(M)]
        else:
            # print(
            #    "The problem does not have an optimal solution. Increasing epsilon to {}".format(
            #        epsilon + 1
            #    )
            # )
            epsilon = epsilon + 1

class robust_semi_sinkhorn(torch.autograd.Function):
    """
    Implementing algorithm 1 from https://proceedings.neurips.cc/paper/2021/file/b80ba73857eed2a36dc7640e2310055a-Paper.pdf
    """

    @staticmethod
    def forward(ctx, cost, eta, r_in, tau, n_iter=500):
        N, c = cost.shape
        a = torch.ones((1, N)).to(cost.device)
        b = (r_in.reshape((1, c)) * N).to(cost.device)
        u, v = torch.zeros((1, N)).to(cost.device), torch.zeros((1, c)).to(cost.device)

        for i in range(n_iter):
            B = torch.exp((u.T + v - cost) / eta)

            if i % 2 == 0:
                a_k = torch.sum(B, dim=1)
                u += eta * (torch.log(a) - torch.log(a_k))
            else:
                b_k = torch.sum(B, dim=0)
                v = eta * tau / (eta + tau) * (v / eta + torch.log(b) - torch.log(b_k))

        p = torch.exp((u.T + v - cost) / eta)
        # ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        # ctx.eta = eta

        return p

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: to be implemented; leave for future work
        return None
