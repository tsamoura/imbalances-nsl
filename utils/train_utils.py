import torch
import shutil
import torch.nn.functional as F
from utils.utils_statistics import accuracy, accuracy_partial, accuracy_partial_cifar


def test_unbiased(acc_shot, model, test_loader, verbose=True, ratio=None, tau=1):
    with torch.no_grad():
        if verbose:
            print("==> Evaluation...")
        model.eval()
        pred_list = []
        true_list = []
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            if ratio is not None:
                outputs = model(images, ratio=ratio, tau=tau)
            else:
                outputs = model(images)
            pred = F.softmax(outputs, dim=1)
            pred_list.append(pred.cpu())
            true_list.append(labels)

        pred_list = torch.cat(pred_list, dim=0)
        true_list = torch.cat(true_list, dim=0)

        acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
        acc_many, acc_med, acc_few, class_accs = acc_shot.get_shot_acc(
            pred_list.max(dim=1)[1], true_list, acc_per_cls=True
        )
        if verbose:
            print(
                "==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]"
                % (acc1, acc5, acc_many, acc_med, acc_few)
            )
            print("==> Class-specific accuracies are", class_accs)
    return float(acc1), float(acc_many), float(acc_med), float(acc_few)


def test_otla(
    acc_shot,
    model,
    test_loader,
    sinkhorn_fn,
    est_ratio,
    lamd,
    tau,
    verbose=True,
):
    with torch.no_grad():
        if verbose:
            print("==> Evaluation...")
        model.eval()
        pred_list = []
        true_list = []
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            pred = model(images, do_softmax=True)
            # if apply_softmax:
            #     pred = F.softmax(pred, dim=1)
            pred = sinkhorn_fn(-torch.log(pred), lamd, est_ratio, tau)
            pred_list.append(pred.cpu())
            true_list.append(labels)

        pred_list = torch.cat(pred_list, dim=0)
        true_list = torch.cat(true_list, dim=0)

        acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
        acc_many, acc_med, acc_few, class_accs = acc_shot.get_shot_acc(
            pred_list.max(dim=1)[1], true_list, acc_per_cls=True
        )
        if verbose:
            print("==> Class-specific accuracies are", class_accs)
    return float(acc1), float(acc_many), float(acc_med), float(acc_few)


def test_partial(model, partial_loader, transition, ratio=None, tau=1):
    with torch.no_grad():
        model.eval()
        num_correct_prediction = 0
        num_data = 0
    
        for i, (xs, _, ss, _) in enumerate(partial_loader):
            xs = [x.cuda() for x in xs]
            outputs = [model(x, ratio, tau) for x in xs]
            _, batch_correct, batch_size = accuracy_partial(outputs, ss, transition)
            num_correct_prediction += batch_correct
            num_data += batch_size

    return num_correct_prediction / num_data


def test_partial_cifar(model, partial_loader, label_vec_to_partial, ratio=None, tau=1):
    with torch.no_grad():
        model.eval()
        num_correct_prediction = 0
        num_data = 0

        transition = lambda l1, l2: label_vec_to_partial[(l1, l2)]
    
        for i, (x1_w, _, x2_w, _, Y1, Y2, y1, y2, s1, s2, _, _) in enumerate(partial_loader):
            true_partial = [transition(int(y1[i]), int(y2[i])) for i in range(len(y1))]

            xs = [x1_w.cuda(), x2_w.cuda()]
            outputs = [model(x, do_softmax=True, ratio=ratio, tau=tau) for x in xs]
            batch_correct, batch_size = accuracy_partial_cifar(outputs, true_partial, transition)
            num_correct_prediction += batch_correct
            num_data += batch_size

    return num_correct_prediction / num_data


def test_partial_otla(
    model, partial_loader, transition, sinkhorn_fn, est_ratio, lamd, tau, epochs=1
):
    with torch.no_grad():
        model.eval()
        num_correct_prediction = 0
        num_data = 0

        for epoch in range(epochs):
            for xs, _, ss, _ in partial_loader:
                xs = [x.cuda() for x in xs]
                outputs = [model(x, do_softmax=True) for x in xs]
                batch_size = outputs[0].size(0)
                outputs = torch.log(torch.cat(outputs, dim=0))
                outputs = sinkhorn_fn(-outputs, lamd, est_ratio, tau)
                outputs = torch.split(outputs, batch_size, dim=0)
                _, batch_correct, batch_size = accuracy_partial(outputs, ss, transition)
                num_correct_prediction += batch_correct
                num_data += batch_size

    return num_correct_prediction / num_data


def test_partial_cifar_otla(
    model, partial_loader, label_vec_to_partial, sinkhorn_fn, est_ratio, lamd, tau, epochs=1
):
    with torch.no_grad():
        model.eval()
        num_correct_prediction = 0
        num_data = 0

        transition = lambda l1, l2: label_vec_to_partial[(l1, l2)]

        for epoch in range(epochs):
            for x1_w, _, x2_w, _, Y1, Y2, y1, y2, s1, s2, _, _ in partial_loader:
                true_partial = [transition(int(y1[i]), int(y2[i])) for i in range(len(y1))]

                xs = [x1_w.cuda(), x2_w.cuda()]
                outputs = [model(x, do_softmax=True) for x in xs]
                batch_size = outputs[0].size(0)
                outputs = torch.log(torch.cat(outputs, dim=0))
                outputs = sinkhorn_fn(-outputs, lamd, est_ratio, tau)
                outputs = torch.split(outputs, batch_size, dim=0)
                batch_correct, batch_size = accuracy_partial_cifar(outputs, true_partial, transition)
                num_correct_prediction += batch_correct
                num_data += batch_size

    return num_correct_prediction / num_data


def test_records(acc_shot, model, test_loader, feat_mean=None, reshape_view=None):
    with torch.no_grad():
        print("==> Evaluation...")
        model.eval()
        pred_list = []
        true_list = []
        if feat_mean is not None:
            if reshape_view is not None:
                bias = model.fc(feat_mean.view(*reshape_view)).detach()
            else:
                bias = model.fc(feat_mean.unsqueeze(0)).detach()
            bias = F.softmax(bias, dim=1)
        for _, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            outputs = model(images, eval_only=True)
            # pred = F.softmax(outputs, dim=1)
            # if feat_mean is not None:
            # pred = F.softmax(outputs - torch.log(bias + 1e-9), dim=1)
            pred = outputs - torch.log(bias + 1e-9)
            pred_list.append(pred.cpu())
            true_list.append(labels)

        pred_list = torch.cat(pred_list, dim=0)
        true_list = torch.cat(true_list, dim=0)

        acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
        acc_many, acc_med, acc_few = acc_shot.get_shot_acc(
            pred_list.max(dim=1)[1], true_list
        )
        print(
            "==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]"
            % (acc1, acc5, acc_many, acc_med, acc_few)
        )
    return float(acc1), float(acc_many), float(acc_med), float(acc_few)


def estimate_empirical_distribution(model, est_loader, num_class):
    with torch.no_grad():
        print("==> Estimating empirical label distribution ...")
        model.eval()
        est_pred_list = []
        for _, (images, labels, _) in enumerate(est_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            pred = torch.softmax(outputs, dim=1) * labels
            est_pred_list.append(pred.cpu())

    est_pred_idx = torch.cat(est_pred_list, dim=0).max(dim=1)[1]
    est_pred = F.one_hot(est_pred_idx, num_class).detach()
    emp_dist = est_pred.sum(0)
    emp_dist = emp_dist / float(emp_dist.sum())

    return emp_dist.unsqueeze(1)


def mipll_estimate_empirical_distribution(model, est_loader, num_class):
    with torch.no_grad():
        print("==> Estimating empirical label distribution ...")
        model.eval()
        est_pred_list = []
        for _, (xs, ys, ss, _) in enumerate(est_loader):
            xs = [x.cuda() for x in xs]
            M = len(xs)
            logits_x = [model(x, do_softmax=True) for x in xs]

            for j in range(M):
                est_pred_list.append(logits_x[j].cpu())

    est_pred_idx = torch.cat(est_pred_list, dim=0).max(dim=1)[1]
    est_pred = F.one_hot(est_pred_idx, num_class).detach()
    emp_dist = est_pred.sum(0)
    emp_dist = emp_dist / float(emp_dist.sum())

    return emp_dist.unsqueeze(1)


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", best_file_name="model_best.pth.tar"
):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)
