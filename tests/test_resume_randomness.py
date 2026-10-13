"""A resumed optimizer update must continue the stochastic model stream."""
import random

import numpy as np
import torch

from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.random_state import capture, restore


def test_rng_round_trip_survives_weights_only_serialization(tmp_path):
    old = capture()
    try:
        saved = capture()
        expected = (random.random(), np.random.rand(3), torch.rand(3))  # noqa: NPY002 -- test legacy global RNG
        path = tmp_path / "rng.pt"
        torch.save(saved, path)
        restore(torch.load(path, weights_only=True))
        actual = (random.random(), np.random.rand(3), torch.rand(3))  # noqa: NPY002
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    finally:
        restore(old)


def test_checkpoint_resume_matches_uninterrupted_dropout_update(tmp_path):
    model = torch.nn.Sequential(torch.nn.Dropout(.5), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
    x = torch.ones(8, 4)

    def update():
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        scheduler.step()

    update()
    checkpoint = save_checkpoint(tmp_path, step=1, model=model, optimizer=optimizer,
                                 scheduler=scheduler, scaler=None, config={})
    update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    torch.rand(100)
    load_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)
    update()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_starting_prior_loader_does_not_consume_model_rng(pd_cfg):
    from src.prior.dataset import build_loader

    state = torch.get_rng_state().clone()
    loader = build_loader(pd_cfg["prior"], "pd", batch_size=2, seed=17, num_workers=0, start_batch=9)
    it = iter(loader)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    assert it is not None
