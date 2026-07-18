import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pan1ni.action import DirectPolicyHead, InverseDynamicsHead
from pan1ni.config import ModelConfig
from pan1ni.data import GoalWindowDataset
from pan1ni.losses import sigreg
from pan1ni.model import GoalConditionedLeWorldModel
from pan1ni.minihack_data import MiniHackPixelGoalDataset
from pan1ni.nld_data import NLDHDF5GoalDataset
from pan1ni.synthetic import make_goal_directed_trajectories, make_synthetic_trajectories
from pan1ni.train import action_step, label_subset_indices, pretrain_step


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.config = ModelConfig(
            latent_dim=32,
            cell_dim=16,
            message_dim=8,
            hidden_dim=32,
            vit_dim=16,
            vit_layers=1,
            vit_heads=4,
            terminal_patch_size=2,
            projector_hidden_dim=64,
            predictor_layers=1,
            predictor_heads=4,
            max_context=4,
            num_actions=8,
            dropout=0.0,
        )
        trajectories = make_synthetic_trajectories(count=3, length=20, height=5, width=8, num_actions=8)
        dataset = GoalWindowDataset(trajectories, context_length=4, samples_per_epoch=8)
        cls.trajectories = trajectories
        cls.dataset = dataset
        cls.batch = next(iter(DataLoader(dataset, batch_size=3)))

    def test_goal_is_final_episode_frame(self):
        item = self.dataset[0]
        trajectory = self.trajectories[item["trajectory_id"].item()]
        for key, value in item["goal"].items():
            self.assertTrue(torch.equal(value, trajectory.observations[key][-1]))
        self.assertEqual(
            item["goal_offset"].item(),
            trajectory.length - 1 - item["timestep"].item(),
        )

    def test_goal_directed_chain_reduces_distance_each_step(self):
        trajectory = make_goal_directed_trajectories(count=1, min_distance=6, seed=4)[0]
        positions = trajectory.observations["cursor"]
        goal = positions[-1]
        distances = (positions - goal).abs().sum(-1)
        self.assertTrue(torch.equal(distances[:-1] - distances[1:], torch.ones_like(distances[:-1])))
        self.assertEqual(trajectory.actions.shape[0], trajectory.length - 1)

    def test_hdf5_sampler_uses_end_of_sequence_as_goal(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "tiny.hdf5")
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("7")
                chars = np.zeros((10, 24, 80), dtype=np.uint8)
                chars[:, 1, 1] = np.arange(10)
                episode.create_dataset("tty_chars", data=chars)
                episode.create_dataset("tty_colors", data=np.zeros_like(chars, dtype=np.int8))
                episode.create_dataset("tty_cursor", data=np.zeros((10, 2), dtype=np.int16))
                episode.create_dataset("actions", data=np.arange(10, dtype=np.int16))
            dataset = NLDHDF5GoalDataset(
                path,
                context_length=1,
                goal_horizon=3,
                samples_per_epoch=1,
            )
            item = dataset[0]
            current_value = item["history"]["chars"][0, 1, 1].item()
            target_value = item["target"]["chars"][1, 1].item()
            goal_value = item["goal"]["chars"][1, 1].item()
            self.assertEqual(target_value, current_value + 1)
            self.assertEqual(goal_value, current_value + 3)

    def test_goal_conditioned_shapes(self):
        model = GoalConditionedLeWorldModel(self.config)
        self.assertTrue(hasattr(model.encoder.backbone, "cls_token"))
        self.assertTrue(any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder.proj.modules()))
        output = model(self.batch["history"], self.batch["goal"])
        self.assertEqual(output.next_latent.shape, (3, 32))
        self.assertEqual(output.history_latents.shape, (3, 4, 32))

    def test_pixel_vit_goal_conditioned_shapes(self):
        config = ModelConfig(
            observation_mode="pixels",
            latent_dim=32,
            hidden_dim=32,
            vit_dim=16,
            vit_layers=1,
            vit_heads=4,
            pixel_patch_size=16,
            max_patches=128,
            projector_hidden_dim=64,
            predictor_layers=1,
            predictor_heads=4,
            max_context=2,
            dropout=0.0,
        )
        model = GoalConditionedLeWorldModel(config)
        batch = {
            "history": {"pixels": torch.randint(256, (3, 1, 3, 144, 144), dtype=torch.uint8)},
            "goal": {"pixels": torch.randint(256, (3, 3, 144, 144), dtype=torch.uint8)},
        }
        output = model(batch["history"], batch["goal"])
        self.assertEqual(output.next_latent.shape, (3, 32))
        self.assertTrue(any(isinstance(layer, nn.BatchNorm1d) for layer in model.encoder.proj.modules()))

    def test_minihack_pixel_sampler_uses_final_goal(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, "pixels.hdf5")
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("0")
                pixels = np.zeros((4, 144, 144, 3), dtype=np.uint8)
                pixels[:, 0, 0, 0] = np.arange(4)
                episode.create_dataset("pixels", data=pixels)
                episode.create_dataset("actions", data=np.arange(3, dtype=np.int16))
                episode.create_dataset("stages", data=np.arange(4, dtype=np.int8))
            dataset = MiniHackPixelGoalDataset(path, samples_per_epoch=1)
            item = dataset[0]
            self.assertEqual(item["goal"]["pixels"][0, 0, 0].item(), 3)
            self.assertEqual(item["target"]["pixels"][0, 0, 0].item(), item["history"]["pixels"][0, 0, 0, 0].item() + 1)
            dataset._handle.close()
            dataset._handle = None

    def test_pretraining_step(self):
        model = GoalConditionedLeWorldModel(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = pretrain_step(model, self.batch, optimizer, sigreg_slices=8)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_label_subsets_are_nested(self):
        small = label_subset_indices(1000, 0.01, seed=7)
        large = label_subset_indices(1000, 0.1, seed=7)
        self.assertTrue(set(small.tolist()).issubset(large.tolist()))

    def test_action_heads(self):
        for head, direct in (
            (InverseDynamicsHead(32, 8, 32), False),
            (DirectPolicyHead(32, 8, 32), True),
        ):
            model = GoalConditionedLeWorldModel(self.config)
            optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
            metrics = action_step(model, head, self.batch, optimizer, direct=direct)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)

    def test_sigreg_detects_collapse(self):
        torch.manual_seed(1)
        gaussian = sigreg(torch.randn(512, 16), num_slices=32)
        collapsed = sigreg(torch.zeros(512, 16), num_slices=32)
        self.assertLess(gaussian, collapsed)


if __name__ == "__main__":
    unittest.main()
