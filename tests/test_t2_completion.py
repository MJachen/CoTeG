import unittest

import torch

from t2_completion import (
    ARM_FULL_COTEG,
    ARM_POPULATION_NO_GRAPH,
    CompletionConfig,
    fit_population_training_basis,
    infer_per_subject_completion,
    infer_population_completion,
    training_capacity,
)


class T2CompletionTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.x = torch.rand(2, 4, 8, 8, 8)
        self.groups = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.cfg = CompletionConfig(
            rank_s=2,
            rank_m=2,
            rank_n=2,
            train_iterations=1,
            eval_iterations=1,
            lr_train=1e-3,
            lr_eval=1e-3,
            lambda_space=1e-3,
            lambda_mod=5e-4,
            lambda_subject=5e-4,
            lambda_tv=0.0,
            tv_max_pairs=0,
            chunk_s=128,
            knn_k=1,
            graph_sigma=1.0,
            seed=7,
            per_subject_batch_size=2,
        )

    def test_population_arms_have_identical_capacity(self):
        shape = tuple(self.x.shape)
        no_graph = training_capacity(ARM_POPULATION_NO_GRAPH, shape, self.cfg)
        full = training_capacity(ARM_FULL_COTEG, shape, self.cfg)
        self.assertEqual(no_graph, full)

    def test_all_backends_restore_observed_modalities_exactly(self):
        basis, _, _ = fit_population_training_basis(
            self.x, self.groups, self.cfg, self.device, use_subject_graph=False
        )
        omega = self.groups[1]
        population, _, _, _ = infer_population_completion(
            basis,
            self.x,
            omega,
            self.cfg,
            self.device,
            use_subject_graph=False,
            seed=17,
        )
        per_subject, _, _, _ = infer_per_subject_completion(
            self.x, omega, self.cfg, self.device, seed=17
        )
        observed = omega.bool()
        self.assertTrue(torch.equal(population[:, observed], self.x[:, observed]))
        self.assertTrue(torch.equal(per_subject[:, observed], self.x[:, observed]))


if __name__ == "__main__":
    unittest.main()
