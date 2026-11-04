import unittest

import numpy as np

import failure_gym as g
import trajectory_gym as t


class TrajectoryRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = {k: v.astype(np.float64)
                       for k, v in np.load(t.ROOT / "best_weights_v4.npz").items()}
        failures, cls.probes = g.mine(cls.weights, [9], use_flight=True)
        if not failures:
            raise RuntimeError("seed 9 no longer provides the expected v4 failure")
        cls.trajectory = t.build_trajectory(cls.weights, failures[0])

    def test_known_rescue_contains_downstream_defect(self):
        self.assertEqual(self.trajectory.root.seed, 9)
        self.assertEqual(self.trajectory.root.step, 17)
        self.assertGreaterEqual(len(self.trajectory.states), 4)
        defective = t.defective_states(self.weights, self.trajectory.states)
        self.assertGreater(len(defective), 1)
        self.assertTrue(any(state.step > self.trajectory.root.step for state in defective))

    def test_original_policy_fails_actual_rescue_replay(self):
        replay = t.replay_rescue(self.weights, self.trajectory)
        self.assertFalse(replay["success"])
        self.assertEqual(replay["reason"], "miss")

    def test_every_forced_state_has_a_saving_action(self):
        self.assertTrue(all(state.saving for state in self.trajectory.states))
        self.assertEqual(len(self.trajectory.states), len(self.trajectory.forced_actions))
        for state, action in zip(self.trajectory.states, self.trajectory.forced_actions):
            self.assertIn(action, state.saving)

    def test_w3_solve_meets_each_linear_constraint_exactly(self):
        active = t.defective_states(self.weights, self.trajectory.states)
        candidate = t.solve_w3_trajectory(self.weights, self.trajectory.states,
                                          self.probes, lam=100.0)
        for state in active:
            save = max(state.saving, key=lambda a: state.logits[a])
            _, old_logits = g.hidden_and_logits(self.weights, state.obs)
            _, new_logits = g.hidden_and_logits(candidate, state.obs)
            old_gap = old_logits[state.wrong] - old_logits[save]
            new_gap = new_logits[state.wrong] - new_logits[save]
            expected = old_gap - (old_gap + t.MARGIN)
            self.assertAlmostEqual(new_gap, expected, places=6)


if __name__ == "__main__":
    unittest.main()
