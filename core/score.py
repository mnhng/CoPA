import sklearn.metrics as metrics
import torch
import torch.nn.functional as F

from .util import copy_to


def _risk_helper(logits, Y, metric):
    metric = metric.lower()
    if metric == 'f1':
        return metrics.f1_score(Y, logits.argmax(dim=1), average='macro')
    elif metric == 'acc':
        return metrics.accuracy_score(Y, logits.argmax(dim=1))

    raise ValueError(metric)


def risk(model, dataset, size, metric, fwd_args=dict()):
    model.eval()
    dev = next(model.parameters()).device
    truth, logits = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=size):
            truth.append(batch.pop('Y'))
            logits.append(model(**copy_to(batch, dev), **fwd_args).cpu())
    truth = torch.cat(truth)
    logits = torch.cat(logits, dim=0)

    if isinstance(metric, list):
        return {m: _risk_helper(logits, truth, m) for m in metric}

    return _risk_helper(logits, truth, metric), truth, logits


def update_chkpnt_(best, model, slot, risk_score, iter_):
    if best['risk_score'][slot] < risk_score:
        best['risk_score'][slot] = risk_score
        best['state_dict'][slot] = copy_to(model.state_dict(), 'cpu')
        best['iter'][slot] = iter_


def validate(model_, best_, envs, batch_size, iteration):
    risks = [risk(model_, E, batch_size, 'F1') for E in envs]
    record = {}
    for i, (r, ytrue, ylogit) in enumerate(risks):
        update_chkpnt_(best_, model_, i, r, iteration)
        record[f'Val {i}th'] = r

    return record


def risk_round(model, environments, batch_size, prefix, fwd_args=dict(), return_prediction=False):
    def _key(*strings):
        return ','.join(strings)

    pred, out = {}, {}
    for i, data in enumerate(environments):
        scores, m_pred = risk_v2(model, data, batch_size, ['f1', 'acc'], fwd_args)
        out.update({_key(prefix, data.E, k): v for k, v in scores.items()})
        pred.update({f'{i}_{k}': v for k, v in m_pred.items()})

    if return_prediction:
        return out, pred

    return out


def risk_v2(model, dataset, size, metric, fwd_args=dict()):
    assert isinstance(metric, list)

    model.eval()
    dev = next(model.parameters()).device
    ids, truth, logits = [], [], []
    Z, prior = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=size):
            ids.extend(batch.pop('id'))
            truth.append(batch.pop('Y'))

            Z.append(batch['Z'])
            p = fwd_args.get('prior')
            prior.append(batch[p if p is not None else 'YZE'])

            logits.append(model(**copy_to(batch, dev), **fwd_args).cpu())

    truth = torch.cat(truth)
    logits = torch.cat(logits, dim=0)

    m_out = {'ids': ids, 'true': truth, 'logit': logits, 'Z': torch.cat(Z), 'P': torch.cat(prior, dim=0)}

    return {m: _risk_helper(logits, truth, m) for m in metric}, m_out


class RiskEvaluator():
    def __init__(self, train_envs, test_envs, batch_size):
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.batch_size = batch_size

    def do(self, model, prefix, fwd_args=dict()):
        def _key(*strings):
            return ','.join(strings)

        metrics = ['f1', 'acc']

        out, labels  = {}, {}
        for data in self.test_envs:
            scores, m_out = risk_v2(model, data, self.batch_size, metrics, fwd_args)
            out |= {_key(prefix, data.E, k): v for k, v in scores.items()}
            labels |= {_key(prefix, data.E, k): v for k, v in m_out.items()}

        return out, labels

    def __call__(self, model, prefix, fwd_args=dict()):
        if prefix == 'end':
            return risk_round(model, self.train_envs, self.batch_size, prefix, fwd_args)

        return risk_round(model, self.test_envs, self.batch_size, prefix, fwd_args)
