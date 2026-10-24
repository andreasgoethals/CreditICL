"""Experiment 2 contracts against the installed, pinned TabPFN fine-tuner.

Tiny logits exercise the actual upstream loss methods and their gradients without
loading checkpoints or training a foundation model. These run in Exp0's CPU suite.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.train.tabpfn_trainer import (
    REGRESSION_LOSS_WEIGHTS,
    TabPFNTrainer,
    classification_loss,
    regression_loss,
)


@pytest.mark.parametrize("strategy", ["scratch", "head", "freeze", "unknown", None])
def test_tabpfn_refuses_unsupported_initialization_before_writing(tmp_path, strategy):
    cfg = {"task": "pd", "prior": {"encoding": "raw"}, "init": {"strategy": strategy}}
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="only init.strategy: full"):
        TabPFNTrainer(cfg, output, device="cpu")
    assert not output.exists()


@pytest.mark.parametrize("alpha", [0.001, -0.001, float("inf"), float("nan")])
def test_tabpfn_refuses_unimplemented_regularization_before_writing(tmp_path, alpha):
    cfg = {"task": "lgd", "prior": {"encoding": "raw"}, "init": {"strategy": "full"},
           "train": {"l2sp_alpha": alpha}}
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="train.l2sp_alpha must be 0"):
        TabPFNTrainer(cfg, output, device="cpu")
    assert not output.exists()


@pytest.mark.parametrize("n_classes", [2, 10])
@pytest.mark.parametrize("n_estimators", [1, 2, 4])
def test_classification_loss_and_gradients_match_official_finetuner(n_classes, n_estimators):
    from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

    generator = torch.Generator().manual_seed(291)
    ours_logits = torch.randn(7, 1, n_estimators, n_classes, generator=generator,
                             requires_grad=True)
    official_logits = ours_logits.detach().clone().requires_grad_()
    batch = SimpleNamespace(X_query=None, y_query=(torch.arange(7) % n_classes)[None])
    estimator = SimpleNamespace(n_classes_=n_classes,
                                forward=lambda _, **kw: ours_logits)
    official = FinetunedTabPFNClassifier(
        device="cpu", n_estimators_finetune=n_estimators,
        n_estimators_validation=n_estimators, n_estimators_final_inference=n_estimators,
    )
    official._local_n_estimators_ = n_estimators
    official.finetuned_estimator_ = estimator
    official._training_forward = lambda _, **kw: official_logits

    loss = classification_loss(estimator, batch, n_estimators)
    expected = official._forward_with_loss(batch)
    gradient, = torch.autograd.grad(loss, ours_logits)
    expected_gradient, = torch.autograd.grad(expected, official_logits)

    assert torch.isfinite(loss) and gradient.abs().sum() > 0
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("transform_member", [False, True])
def test_regression_loss_and_gradients_match_official_finetuner(transform_member):
    from sklearn.preprocessing import PowerTransformer
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    n_estimators, n_queries, n_bars = 2, 7, 6
    generator = torch.Generator().manual_seed(293)
    ours_logits = torch.randn(n_queries, 1, n_estimators, n_bars, generator=generator,
                             requires_grad=True)
    official_logits = ours_logits.detach().clone().requires_grad_()
    bar = FullSupportBarDistribution(torch.linspace(-3, 3, n_bars + 1))
    # Fit only on context targets, as the official preprocessing pipeline does.
    target_transform = (PowerTransformer().fit(np.array([-2., -1., -.5, 0., .2, 2., 8.])[:, None])
                        if transform_member else None)
    batch = SimpleNamespace(
        X_query=None, y_query=torch.tensor([[-5., -2., -.2, 0., .8, 2., 7.]]),
        znorm_space_bardist=bar,
        configs=[[SimpleNamespace(target_transform=None)],
                 [SimpleNamespace(target_transform=target_transform)]],
    )
    estimator = SimpleNamespace(forward=lambda _: (None, list(ours_logits.unbind(2)), None))
    official = FinetunedTabPFNRegressor(
        device="cpu", n_estimators_finetune=n_estimators,
        n_estimators_validation=n_estimators, n_estimators_final_inference=n_estimators,
    )
    official._local_n_estimators_ = n_estimators
    official._bardist_loss = bar
    official._training_forward = lambda _: (None, list(official_logits.unbind(2)), None)

    loss = regression_loss(estimator, batch, n_estimators, REGRESSION_LOSS_WEIGHTS)
    expected = official._forward_with_loss(batch)
    gradient, = torch.autograd.grad(loss, ours_logits)
    expected_gradient, = torch.autograd.grad(expected, official_logits)

    assert torch.isfinite(loss) and gradient.abs().sum() > 0
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    if transform_member:
        # The fixture must detect accidentally applying one common target space
        # to both members; numerical equality alone is not useful without this.
        common_batch = SimpleNamespace(**{**vars(batch), "configs": [batch.configs[0]] * 2})
        wrong_loss = regression_loss(estimator, common_batch, n_estimators, REGRESSION_LOSS_WEIGHTS)
        assert not torch.isclose(loss, wrong_loss)
