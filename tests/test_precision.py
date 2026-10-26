"""Policy and table tests; run without checkpoint loading or third-party test tools."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "app/inference/precision.py"
spec = importlib.util.spec_from_file_location("precision_policy", MODULE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
CLASSES = [
    "Anthracnose",
    "Corticium",
    "Mealy",
    "Miner",
    "Nematode",
    "Phoma",
    "Rust",
    "Healthy",
]


def branch(label, confidence, secondary="Mealy"):
    probs = [0.01] * len(CLASSES)
    probs[CLASSES.index(label)] = confidence
    if secondary != label:
        probs[CLASSES.index(secondary)] = 0.05
    return {"pred_class": label, "pred_conf": confidence, "probs": probs}


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.table = policy.load_precision_table(
            MODULE.parent / "data/precision_by_class.csv", CLASSES
        )

    def test_agreement_selects_higher_confidence_and_its_secondary(self):
        big, small = branch("Healthy", 0.7), branch("Healthy", 0.9, "Rust")
        result = policy.select_by_precision(big, small, CLASSES, self.table)
        self.assertEqual(
            (result["pred_class"], result["confidence"], result["selected_branch"]),
            ("Healthy", 0.9, "small"),
        )
        self.assertEqual(result["top2_class"], "Rust")
        self.assertEqual(result["method"], "agreement")
        self.assertEqual(big["pred_class"], "Healthy")

    def test_agreement_confidence_tie_chooses_big(self):
        result = policy.select_by_precision(
            branch("Rust", 0.8), branch("Rust", 0.8), CLASSES, self.table
        )
        self.assertEqual(result["selected_branch"], "big")

    def test_small_precision_overrides_higher_big_confidence(self):
        result = policy.select_by_precision(
            branch("Nematode", 0.99), branch("Rust", 0.6), CLASSES, self.table
        )
        self.assertEqual(
            (result["pred_class"], result["confidence"], result["method"]),
            ("Rust", 0.6, "precision_small"),
        )
        self.assertEqual(result["top2_class"], "Nematode")
        self.assertEqual(result["top2_confidence"], 0.99)

    def test_big_uses_precision_of_its_predicted_class(self):
        result = policy.select_by_precision(
            branch("Corticium", 0.6), branch("Mealy", 0.99), CLASSES, self.table
        )
        self.assertEqual(result["method"], "precision_big")
        self.assertEqual(result["pred_class"], "Corticium")

    def test_equal_precision_selects_big_even_with_lower_confidence(self):
        result = policy.select_by_precision(
            branch("Phoma", 0.51), branch("Miner", 0.99), CLASSES, self.table
        )
        self.assertEqual(result["method"], "precision_tie_big")
        self.assertEqual(result["pred_class"], "Phoma")

    def test_invalid_tables_fail(self):
        cases = [
            "class,precision_big,precision_small\nA,nan,.5\n",
            "class,precision_big,precision_small\nA,1.1,.5\n",
            "class,precision_big,precision_small\nA,.5,.5\nA,.6,.6\n",
            "class,precision_big,precision_small\nB,.5,.5\n",
            "class,precision_big\nA,.5\n",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.csv"
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        policy.load_precision_table(path, ["A"])

    def test_missing_table_fails(self):
        with self.assertRaises(FileNotFoundError):
            policy.load_precision_table(
                MODULE.parent / "missing_precision.csv", CLASSES
            )


if __name__ == "__main__":
    unittest.main()
