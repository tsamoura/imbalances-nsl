import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def evaluate_disjunction_tensor(
    lineage: torch.tensor, softmax_predictions: torch.tensor
) -> torch.tensor:
    # find all positions of 1's
    indices = torch.nonzero(lineage, as_tuple=False)
    # get the entries of those positions
    predictions = softmax_predictions[indices]
    # 1 - those entries
    predictions = 1 - predictions
    return 1 - torch.prod(predictions)

# The proofs are of the form X_{1,superclassA} \wedge X_{2,superclassB} \vee X_{1,superclassB} \wedge X_{2,superclassA},
# where superclassA <> superclassB.
# For these kind of proofs, WMC is computed as follows:
# 1 - (1- P(y1 \in superclassA and y2 \in superclassB)) (1- P(y1 \in superclassB and y1 \in superclassA)) =
# 1 - (1- P(y1 \in superclassA) * P(y2 \in superclassB)) (1- P(y1 \in superclassB) * P(y1 \in superclassA)),
# where P(y \in superclassY) = 1 - \prod_{y_i \in superclassY} (1-P(y = y_i)).
# We allow superclassA = superclassB, only when superclassA is a base class.
# This is because, in the case of overlaps, the formula to compute WMC becomes more complex.
# If superclassA = superclassB, then the loss would be P(y1 \in superclassA) * P(y2 \in superclassB)
class wmc_loss(nn.Module):
    def __init__(self, m=0.9):
        super().__init__()
        self.feat_mean = None
        self.m = m

    def _compute_probability(
        self, lineage1, lineage2, y_pred_1_probas, y_pred_2_probas
    ):
        Px1_Y1 = evaluate_disjunction_tensor(lineage1, y_pred_1_probas)
        Px1_Y2 = evaluate_disjunction_tensor(lineage2, y_pred_1_probas)
        Px2_Y1 = evaluate_disjunction_tensor(lineage1, y_pred_2_probas)
        Px2_Y2 = evaluate_disjunction_tensor(lineage2, y_pred_2_probas)
        probability = 1 - (1 - Px1_Y1 * Px2_Y2) * (1 - Px1_Y2 * Px2_Y1)
        return probability

    def forward(
        self,
        scores1,
        Y1,
        scores2=None,
        Y2=None,
        s1=None,
        s2=None,
        is_logit=True,
        model=None,
        feat1=None,
        feat2=None,
    ):
        if model is not None:
            assert is_logit == True
            assert feat1 is not None
            if Y2 is not None:
                assert feat2 is not None

        # Standard WMC
        if model is None:
            if is_logit:
                y_pred_1_probas = torch.softmax(scores1, dim=1)
                if scores2 is not None:
                    y_pred_2_probas = torch.softmax(scores2, dim=1)
            else:
                y_pred_1_probas = scores1
                if scores2 is not None:
                    y_pred_2_probas = scores2
        else:
            # RECORDS
            if feat2 is None:
                if self.feat_mean is None:
                    self.feat_mean = (1 - self.m) * feat1.detach().mean(0)
                else:
                    self.feat_mean = self.m * self.feat_mean + (
                        1 - self.m
                    ) * feat1.detach().mean(0)
            else:
                if self.feat_mean is None:
                    self.feat_mean = (1 - self.m) * ((feat1 + feat2) / 2).detach().mean(
                        0
                    )
                else:
                    self.feat_mean = self.m * self.feat_mean + (1 - self.m) * (
                        (feat1 + feat2) / 2
                    ).detach().mean(0)

            bias = model.fc(self.feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
            logits = scores1 - torch.log(bias + 1e-9)
            y_pred_1_probas = F.softmax(logits, dim=1)
            if feat2 is not None:
                logits = scores2 - torch.log(bias + 1e-9)
                y_pred_2_probas = F.softmax(logits, dim=1)

        K = scores1.shape[0]
        num_class = scores1.shape[1]
        loss_vec = torch.zeros(K).cuda()
        for i in range(K):
            # This is multi-instance PLL for M = 1
            if scores2 is None and Y2 is None:
                lineage1 = Y1[i, :]
                Px1_Y1 = evaluate_disjunction_tensor(
                    lineage1,
                )

                if Px1_Y1 > 9.2885e-30:
                    loss_vec[i] = -torch.log(Px1_Y1)
            # This is multi-instance PLL for M = 2
            else:
                lineage1 = Y1[i, :]
                lineage2 = Y2[i, :]
                # If classA = classB
                # Notice that we allow this case only when we have base classes
                if (
                    np.array_equal(lineage1.cpu().numpy(), lineage2.cpu().numpy())
                    and torch.sum(lineage1) == 1
                ):
                    indices = torch.nonzero(lineage1, as_tuple=False)
                    sp = y_pred_1_probas[i, :]
                    probability1 = sp[indices]
                    if probability1 > 9.2885e-30:
                        loss_vec[i] = -torch.log(probability1)

                    indices = torch.nonzero(lineage2, as_tuple=False)
                    sp = y_pred_2_probas[i, :]
                    probability2 = sp[indices]
                    if probability2 > 9.2885e-30:
                        loss_vec[i] = loss_vec[i] - torch.log(probability2)
                elif (
                    np.array_equal(lineage1.cpu().numpy(), lineage2.cpu().numpy())
                    and torch.sum(lineage1) == 2
                ):
                    # In CIFAR10, this can only happen, when the base classes have exactly the same parent.
                    # For instance:
                    # home_land_animals <= cat, dog
                    # wild_land_animals <= deer, horse
                    # other_animals <= bird, frog
                    # other_transportation <= airplane, ship
                    # If there are only two base classes, e.g. as in other_animals <= bird, frog and other_transportation <= airplane, ship,
                    # then, do the trick from below, to compute the WMC

                    indices = torch.nonzero(lineage1, as_tuple=True)[0]
                    position1 = indices[0]
                    position2 = indices[1]
                    l1 = torch.zeros(num_class).cuda()
                    l2 = torch.zeros(num_class).cuda()
                    l1[position1] = 1
                    l2[position2] = 1

                    probability = self._compute_probability(
                        l1, l2, y_pred_1_probas[i, :], y_pred_2_probas[i, :]
                    )

                    if probability > 9.2885e-30:
                        loss_vec[i] = -torch.log(probability)
                elif (
                    np.sum(np.multiply(lineage1.cpu().numpy(), lineage2.cpu().numpy()))
                    > 0
                ):
                    raise ValueError(
                        "This case is not supported due to assumptions on the benchmark, i.e., that there should be two preconditions per implication."
                    )
                else:
                    probability = self._compute_probability(
                        lineage1, lineage2, y_pred_1_probas[i, :], y_pred_2_probas[i, :]
                    )
                    if probability > 9.2885e-30:
                        loss_vec[i] = -torch.log(probability)
        return loss_vec
