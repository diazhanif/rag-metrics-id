import json
import math
import subprocess
import sys
import unittest

from rag_metrics_id import evaluate


class EvaluateTests(unittest.TestCase):
    def test_known_metrics_and_macro_average(self):
        report = evaluate([
            {"query": "private query one", "relevant": ["A", "C"], "retrieved": ["C", "B", "A"]},
            {"query": "private query two", "relevant": ["X"], "retrieved": ["Y"]},
        ], k=3)
        first = report["per_query"][0]
        self.assertAlmostEqual(first["precision_at_k"], 2 / 3, places=6)
        self.assertEqual(first["recall_at_k"], 1.0)
        self.assertEqual(first["mrr_at_k"], 1.0)
        ideal_dcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(first["ndcg_at_k"], 1.5 / ideal_dcg, places=6)
        self.assertEqual(report["macro_average"]["recall_at_k"], 0.5)
        self.assertEqual(report["macro_average"]["hit_rate_at_k"], 0.5)

    def test_precision_uses_k_even_when_fewer_results_are_returned(self):
        report = evaluate([{"relevant": ["A"], "retrieved": ["A"]}], k=5)
        self.assertEqual(report["macro_average"]["precision_at_k"], 0.2)
        self.assertEqual(report["macro_average"]["recall_at_k"], 1.0)

    def test_rank_position_affects_mrr_and_ndcg(self):
        report = evaluate([{"relevant": ["A"], "retrieved": ["X", "A"]}], k=3)
        metrics = report["per_query"][0]
        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertEqual(metrics["mrr_at_k"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_k"], 1 / math.log2(3), places=6)

    def test_empty_retrieval_is_valid_but_has_zero_scores(self):
        report = evaluate([{"relevant": ["A"], "retrieved": []}], k=3)
        metrics = report["per_query"][0]
        self.assertEqual(metrics["precision_at_k"], 0.0)
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertIsNone(metrics["first_relevant_rank"])

    def test_invalid_dataset_and_duplicate_ids_are_rejected(self):
        invalid_datasets = [
            [],
            [{"relevant": [], "retrieved": []}],
            [{"relevant": ["A", "A"], "retrieved": []}],
            [{"relevant": ["A"], "retrieved": ["B", "B"]}],
            [{"relevant": [" "], "retrieved": []}],
        ]
        for dataset in invalid_datasets:
            with self.subTest(dataset=dataset):
                with self.assertRaises(ValueError):
                    evaluate(dataset)
        with self.assertRaises(ValueError):
            evaluate([{"relevant": ["A"], "retrieved": []}], k=0)

    def test_report_does_not_echo_queries_or_document_ids(self):
        report = evaluate([{
            "query": "rahasia_user_871",
            "relevant": ["private-doc-45"],
            "retrieved": ["private-doc-45"],
        }])
        serialized = json.dumps(report)
        self.assertNotIn("rahasia_user_871", serialized)
        self.assertNotIn("private-doc-45", serialized)

    def test_cli_reads_json_and_returns_metrics_only(self):
        records = [{"query": "contoh", "relevant": ["d1"], "retrieved": ["d1"]}]
        process = subprocess.run(
            [sys.executable, "-m", "rag_metrics_id", "--k", "2"],
            input=json.dumps(records),
            text=True,
            capture_output=True,
            check=True,
        )
        report = json.loads(process.stdout)
        self.assertEqual(report["k"], 2)
        self.assertEqual(report["macro_average"]["precision_at_k"], 0.5)
        self.assertNotIn("contoh", process.stdout)

    def test_cli_rejects_invalid_input_without_echoing_it(self):
        process = subprocess.run(
            [sys.executable, "-m", "rag_metrics_id"],
            input="teks rahasia bukan JSON",
            text=True,
            capture_output=True,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertNotIn("teks rahasia", process.stderr)


if __name__ == "__main__":
    unittest.main()
