"""Regression checks for the warning fixes; run with unittest discovery."""

import ast
import runpy
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

import rl_utils
from envs.combat import Combat

ROOT = Path(__file__).resolve().parents[1]


def definitions(filename):
    """Load tutorial definitions without launching the long training experiment."""
    source = ast.parse((ROOT / filename).read_text())
    source.body = [
        node
        for node in source.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
    ]
    with tempfile.TemporaryDirectory() as directory:
        module = Path(directory) / filename
        module.write_text(ast.unparse(source))
        return runpy.run_path(str(module), init_globals={"device": torch.device("cpu")})


class WarningRegressions(unittest.TestCase):
    def setUp(self):
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("error")
        torch.set_num_threads(1)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def test_tensor_batches_accept_numpy_observations(self):
        ns = definitions("09_Actor-Critic.py")
        agent = ns["ActorCritic"](4, 8, 2, 1e-3, 1e-3, 0.9, torch.device("cpu"))
        states = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
        self.assertIn(agent.take_action(states[0]), [0, 1])
        agent.update(
            {
                "states": states,
                "actions": [0, 1],
                "rewards": [1.0, 1.0],
                "next_states": states[::-1],
                "dones": [False, True],
            }
        )

    def test_trpo_actions_do_not_construct_tensors_from_array_lists(self):
        import gymnasium as gym

        ns = definitions("10_TRPO.py")
        agent = ns["TRPO"](
            8,
            gym.spaces.Box(-1.0, 1.0, shape=(4,)),
            gym.spaces.Discrete(2),
            0.9,
            0.0005,
            0.5,
            1e-3,
            0.9,
            torch.device("cpu"),
        )
        self.assertIn(agent.take_action(np.zeros(4, dtype=np.float32)), [0, 1])
        torch.testing.assert_close(
            ns["compute_advantage"](0.9, 0.8, torch.tensor([[1.0], [2.0]])),
            torch.tensor([[2.44], [2.0]]),
        )

    def test_dueling_dqn_keeps_public_variant_names(self):
        ns = definitions("07_improved_DQN.py")
        agent = ns["DuelingDQN"](
            3,
            8,
            5,
            1e-3,
            0.9,
            0.1,
            10,
            torch.device("cpu"),
            "DuelingDQN",
        )
        self.assertIsInstance(agent.q_net, ns["VAnet"])
        self.assertEqual(agent.dqn_type, "DuelingDQN")

    def test_multi_agent_replay_preserves_different_observation_sizes(self):
        buffer = rl_utils.ReplayBuffer(4, multi_agent=True)
        state = [np.zeros(8), np.zeros(10), np.zeros(10)]
        buffer.add(state, [np.zeros(5)] * 3, [0.0] * 3, state, [False] * 3)
        states, _, _, next_states, _ = buffer.sample(1)
        self.assertEqual([len(x) for x in states[0]], [8, 10, 10])
        self.assertEqual([len(x) for x in next_states[0]], [8, 10, 10])

    def test_advantage_preserves_shape_and_device(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            delta = torch.tensor([[1.0], [2.0]], device=device, requires_grad=True)
            actual = rl_utils.compute_advantage(0.9, 0.8, delta)
            torch.testing.assert_close(
                actual, torch.tensor([[2.44], [2.0]], device=device)
            )
            self.assertFalse(actual.requires_grad)

    def test_ensemble_optimization_does_not_override_module_train(self):
        for filename in ("15_PETS.py", "16_MBPO.py"):
            ns = definitions(filename)
            kwargs = {"model_alpha": 0.01} if filename == "16_MBPO.py" else {}
            model = ns["EnsembleModel"](3, 1, ensemble_size=2, **kwargs)
            self.assertIs(model.train(False), model)
            self.assertFalse(model.training)
            model.train()
            mean, logvar = model(torch.zeros(2, 4, 4), return_log_var=True)
            loss, _ = model.loss(mean, logvar, torch.zeros(2, 4, 4))
            model.optimize(loss)
            self.assertTrue(torch.isfinite(loss).item())

    def test_combat_with_unequal_team_sizes(self):
        env = Combat(n_agents=1, n_opponents=2, max_steps=2)
        obs = env.reset()
        self.assertEqual(len(obs), 1)
        self.assertEqual(len(env.opp_pos), 2)
        self.assertEqual(env.observation_space[0].dtype, np.dtype("float32"))
        env.step([4])
        _, _, done, info = env.step([4])
        self.assertEqual(done, [True])
        self.assertIn("win", info)
        env.close()

    def test_particle_env_and_gumbel_on_cpu_and_cuda(self):
        ns = definitions("20_MADDPG.py")
        env = ns["make_env"]("simple_adversary")
        self.addCleanup(env.close)
        states = env.reset()
        self.assertEqual([len(s) for s in states], [8, 10, 10])
        actions = [np.eye(5, dtype=np.float32)[0] for _ in states]
        done = [False] * 3
        for _ in range(25):
            obs, rewards, done, _ = env.step(actions)
            self.assertEqual(len(rewards), 3)
            self.assertEqual([len(s) for s in obs], [8, 10, 10])
        self.assertTrue(all(done))
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            logits = torch.zeros(2, 5, device=device, requires_grad=True)
            sample = ns["gumbel_softmax"](logits)
            self.assertTrue(torch.isfinite(sample).all().item())
            self.assertEqual(
                sample.device,
                device if device.type == "cpu" else torch.device("cuda:0"),
            )
            sample.sum().backward()


if __name__ == "__main__":
    unittest.main()
