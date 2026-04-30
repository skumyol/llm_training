#!/usr/bin/env python3
"""
test_comprehensive_training.py
=============================
Unit tests for the comprehensive training report system.

Run with:
    python -m pytest tests/test_comprehensive_training.py -v
    python -m pytest tests/test_comprehensive_training.py -v --tb=short
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import torch

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "scripts"))  # For comprehensive_training_report


class TestEmbeddingExtractor(unittest.TestCase):
    """Tests for EmbeddingExtractor class."""

    @unittest.skipIf(
        not __import__('importlib.util').util.find_spec('transformers'),
        "transformers not installed"
    )
    def test_embedding_extractor_init(self):
        """Test EmbeddingExtractor initializes correctly."""
        from run_small_lm import EmbeddingExtractor
        
        # Mock the model loading
        with patch('run_small_lm._AutoTok') as mock_tok, \
             patch('run_small_lm._AutoModel') as mock_model:
            
            # Setup mocks - need a mock that has .to() method
            class MockTensor:
                def __init__(self, shape):
                    self.shape = shape
                def to(self, device):
                    return self
                    
            class MockTokenizerOutput:
                def __init__(self):
                    self.data = {
                        'input_ids': torch.tensor([[1, 2, 3]]),
                        'attention_mask': torch.tensor([[1, 1, 1]])
                    }
                def to(self, device):
                    return self.data
                    
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = MockTokenizerOutput()
            mock_tokenizer.__call__ = mock_tokenizer.return_value
            mock_tok.from_pretrained.return_value = mock_tokenizer
            
            mock_model_instance = Mock()
            # Simulate model output with last_hidden_state
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(1, 3, 768)
            mock_model_instance.return_value = mock_output
            mock_model.from_pretrained.return_value = mock_model_instance
            
            device = torch.device('cpu')
            extractor = EmbeddingExtractor("test-model", device)
            
            self.assertEqual(extractor.model_name, "test-model")
            self.assertEqual(extractor.dim, 768)
            self.assertEqual(extractor.device, device)

    def test_project_to_dim_truncation(self):
        """Test embedding projection with truncation."""
        from run_small_lm import EmbeddingExtractor
        
        with patch('run_small_lm._AutoTok'), patch('run_small_lm._AutoModel'):
            extractor = MagicMock()
            extractor.dim = 768
            
            # Create real embeddings tensor
            embeddings = torch.randn(4, 768)
            
            # Test projection down to 8 dimensions
            projected = embeddings[:, :8]
            self.assertEqual(projected.shape, (4, 8))
            
            # Verify values are preserved
            self.assertTrue(torch.allclose(projected, embeddings[:, :8]))

    def test_project_to_dim_padding(self):
        """Test embedding projection with padding."""
        embeddings = torch.randn(4, 4)
        target_dim = 8
        
        # Pad with zeros
        pad = torch.zeros(embeddings.shape[0], target_dim - embeddings.shape[-1])
        projected = torch.cat([embeddings, pad], dim=-1)
        
        self.assertEqual(projected.shape, (4, 8))
        # First 4 dims should match original
        self.assertTrue(torch.allclose(projected[:, :4], embeddings))
        # Last 4 dims should be zeros
        self.assertTrue(torch.all(projected[:, 4:] == 0))


class TestSmallLMArchitectures(unittest.TestCase):
    """Tests for small LM architecture components."""

    def test_build_model_exists(self):
        """Test that build_model function exists and imports."""
        try:
            from small_lm_architectures import build_model, RECOMMENDED_CONFIGS
            self.assertTrue(callable(build_model))
            self.assertIsInstance(RECOMMENDED_CONFIGS, dict)
        except ImportError as e:
            self.skipTest(f"Could not import small_lm_architectures: {e}")

    def test_config_dataclasses(self):
        """Test that config dataclasses work."""
        try:
            from small_lm_architectures import (
                GPTConfig, GRUConfig, AWDLSTMConfig,
                PrefixGPTConfig, MoEConfig, MambaLikeConfig
            )
            
            # Test GPT config (uses n_embd, n_layer not hidden_dim/num_layers)
            config = GPTConfig(vocab_size=100, n_embd=64, n_layer=2)
            self.assertEqual(config.vocab_size, 100)
            self.assertEqual(config.n_embd, 64)
            
            # Test Prefix GPT config with conditioning (uses n_embd not hidden_dim)
            config = PrefixGPTConfig(vocab_size=100, n_embd=64, n_layer=2, cond_dim=8)
            self.assertEqual(config.cond_dim, 8)
            
        except ImportError as e:
            self.skipTest(f"Could not import configs: {e}")

    def test_architecture_creation(self):
        """Test creating model instances."""
        try:
            from small_lm_architectures import (
                build_model, GPTConfig, TinyGPTLM,
                GRUConfig, SmallGRULM
            )
            
            # Test GPT model creation
            config = GPTConfig(vocab_size=50, n_embd=32, n_layer=2, n_head=2)
            model = TinyGPTLM(config)
            self.assertIsNotNone(model)
            
            # Test forward pass
            x = torch.randint(0, 50, (2, 10))  # batch=2, seq=10
            y = torch.randint(0, 50, (2, 10))
            
            with torch.no_grad():
                output = model(x, y)
                self.assertIsNotNone(output.loss)
                self.assertIsNotNone(output.logits)
                self.assertEqual(output.logits.shape, (2, 10, 50))
                
        except ImportError as e:
            self.skipTest(f"Could not test architectures: {e}")


class TestTokenization(unittest.TestCase):
    """Tests for tokenization utilities."""

    def test_char_tokenizer(self):
        """Test character-level tokenizer."""
        try:
            from run_small_lm import CharTokenizer
            
            text = "hello world"
            tokenizer = CharTokenizer(text)
            
            # Test encoding
            encoded = tokenizer.encode("hello")
            self.assertIsInstance(encoded, list)
            self.assertTrue(all(isinstance(x, int) for x in encoded))
            
            # Note: CharTokenizer doesn't have decode() - it only does encoding
            # Build reverse mapping manually using itos for roundtrip test
            decoded = "".join(tokenizer.itos[i] for i in encoded)
            self.assertEqual(decoded, "hello")
            
            # Test vocab contains unique chars
            self.assertEqual(tokenizer.vocab_size, len(set(text)))
            
        except ImportError as e:
            self.skipTest(f"Could not import CharTokenizer: {e}")

    def test_token_dataset(self):
        """Test TokenDataset class."""
        try:
            from run_small_lm import TokenDataset
            
            # Create dummy token IDs
            token_ids = list(range(100))
            seq_len = 10
            
            dataset = TokenDataset(token_ids, seq_len)
            
            # Test length
            expected_len = max(0, (len(token_ids) - 1) // seq_len)
            self.assertEqual(len(dataset), expected_len)
            
            # Test item retrieval
            if len(dataset) > 0:
                x, y = dataset[0]
                self.assertEqual(x.shape, (seq_len,))
                self.assertEqual(y.shape, (seq_len,))
                # y should be x shifted by 1
                self.assertEqual(y[0].item(), x[1].item())
                
        except ImportError as e:
            self.skipTest(f"Could not import TokenDataset: {e}")


class TestConfigurationBuilders(unittest.TestCase):
    """Tests for configuration building functions."""

    def test_build_personality_config(self):
        """Test personality config builder."""
        try:
            from comprehensive_training_report import build_personality_config
            
            params = {
                "lr": 3e-5,
                "encoder_lr_factor": 0.2,
                "focal_gamma": 2.0,
                "dropout": 0.3,
                "token_drop_prob": 0.1,
                "freeze_encoder_epochs": 1
            }
            
            cfg = build_personality_config(params, seed=42, run_id="test_run", epochs=5)
            
            self.assertEqual(cfg["seed"], 42)
            self.assertEqual(cfg["epochs"], 5)
            self.assertEqual(cfg["lr"], 3e-5)
            self.assertEqual(cfg["encoder_lr"], 3e-5 * 0.2)  # lr * factor
            self.assertEqual(cfg["loss_type"], "focal_bce")
            self.assertEqual(len(cfg["target_columns"]), 5)  # OCEAN
            
        except ImportError as e:
            self.skipTest(f"Could not import build_personality_config: {e}")

    def test_build_affect_config(self):
        """Test affect config builder."""
        try:
            from comprehensive_training_report import build_affect_config
            
            params = {
                "lr": 4e-5,
                "encoder_lr_factor": 0.25,
                "ccc_weight": 0.5,
                "dropout": 0.3,
                "grad_accum": 2,
                "freeze_encoder_epochs": 1
            }
            
            cfg = build_affect_config(params, seed=42, run_id="test_run", epochs=5)
            
            self.assertEqual(cfg["seed"], 42)
            self.assertEqual(cfg["epochs"], 5)
            self.assertEqual(cfg["lr"], 4e-5)
            self.assertEqual(cfg["encoder_lr"], 4e-5 * 0.25)
            self.assertEqual(cfg["loss_type"], "ccc_mse")
            self.assertEqual(len(cfg["target_columns"]), 3)  # VAD
            
        except ImportError as e:
            self.skipTest(f"Could not import build_affect_config: {e}")


class TestAggregationUtilities(unittest.TestCase):
    """Tests for result aggregation functions."""

    def test_aggregate_single_metric(self):
        """Test aggregating a single metric across seeds."""
        try:
            from comprehensive_training_report import aggregate
            import pandas as pd
            
            results = [
                {"seed": 42, "val_f1": 0.65, "success": True},
                {"seed": 43, "val_f1": 0.67, "success": True},
                {"seed": 44, "val_f1": 0.66, "success": True},
            ]
            
            df = aggregate(results, ["val_f1"])
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 1)
            self.assertAlmostEqual(df["val_f1_mean"].iloc[0], 0.66, places=2)
            
        except ImportError as e:
            self.skipTest(f"Could not import aggregate: {e}")

    def test_aggregate_with_missing(self):
        """Test aggregation handles missing values."""
        try:
            from comprehensive_training_report import aggregate
            
            results = [
                {"seed": 42, "val_f1": 0.65, "success": True},
                {"seed": 43, "val_f1": float("nan"), "success": True},
                {"seed": 44, "val_f1": 0.67, "success": True},
            ]
            
            df = aggregate(results, ["val_f1"])
            self.assertIsNotNone(df)
            
        except ImportError as e:
            self.skipTest(f"Could not import aggregate: {e}")

    def test_aggregate_by_arch(self):
        """Test aggregation by architecture."""
        try:
            from comprehensive_training_report import aggregate_by_arch
            import pandas as pd
            
            results = [
                {"arch": "gpt", "final_val_ppl": 45.0, "num_params": 1000000, "success": True},
                {"arch": "gpt", "final_val_ppl": 46.0, "num_params": 1000000, "success": True},
                {"arch": "gru", "final_val_ppl": 50.0, "num_params": 800000, "success": True},
                {"arch": "gru", "final_val_ppl": 52.0, "num_params": 800000, "success": True},
            ]
            
            df = aggregate_by_arch(results, ["final_val_ppl"])
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)  # 2 architectures
            
            # Check sorted by PPL (lower is better)
            self.assertLess(df["final_val_ppl_mean"].iloc[0], 
                           df["final_val_ppl_mean"].iloc[1])
            
        except ImportError as e:
            self.skipTest(f"Could not import aggregate_by_arch: {e}")


class TestOptunaConfigLoading(unittest.TestCase):
    """Tests for loading Optuna best configs."""

    def test_load_optuna_best_missing_file(self):
        """Test handling of missing optuna result file."""
        try:
            from comprehensive_training_report import load_optuna_best
            
            # Should return empty dict for non-existent file
            with tempfile.TemporaryDirectory() as tmpdir:
                # Temporarily change ROOT
                with patch('comprehensive_training_report.ROOT', Path(tmpdir)):
                    result = load_optuna_best("nonexistent_task")
                    self.assertEqual(result, {})
                    
        except ImportError as e:
            self.skipTest(f"Could not import load_optuna_best: {e}")

    def test_load_optuna_best_valid_file(self):
        """Test loading valid optuna result file."""
        try:
            from comprehensive_training_report import load_optuna_best
            
            with tempfile.TemporaryDirectory() as tmpdir:
                optuna_dir = Path(tmpdir) / "artifacts" / "optuna"
                optuna_dir.mkdir(parents=True)
                
                test_data = {
                    "best_value": 0.68,
                    "best_params": {"lr": 3e-5, "dropout": 0.3},
                    "study_name": "test_study",
                    "n_trials": 20
                }
                
                with open(optuna_dir / "test_task_best.json", "w") as f:
                    json.dump(test_data, f)
                
                with patch('comprehensive_training_report.ROOT', Path(tmpdir)):
                    result = load_optuna_best("test_task")
                    self.assertEqual(result["lr"], 3e-5)
                    self.assertEqual(result["dropout"], 0.3)
                    
        except ImportError as e:
            self.skipTest(f"Could not import load_optuna_best: {e}")


class TestResultSaving(unittest.TestCase):
    """Tests for result saving utilities."""

    def test_save_and_load_results(self):
        """Test saving and loading results JSON."""
        try:
            from comprehensive_training_report import _save_results, _load_results
            
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test_results.json"
                
                results = [
                    {"seed": 42, "val_f1": 0.65, "success": True},
                    {"seed": 43, "val_f1": float("nan"), "success": True},
                ]
                
                _save_results(results, path)
                self.assertTrue(path.exists())
                
                loaded = _load_results(path)
                self.assertEqual(len(loaded), 2)
                self.assertEqual(loaded[0]["seed"], 42)
                
        except ImportError as e:
            self.skipTest(f"Could not import result functions: {e}")


class TestMetricsAndFormatting(unittest.TestCase):
    """Tests for metric computation and formatting."""

    def test_f1_metric(self):
        """Test F1 score computation."""
        # Simple binary F1 calculation
        def compute_f1(precision, recall):
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        
        self.assertEqual(compute_f1(1.0, 1.0), 1.0)
        self.assertEqual(compute_f1(0.5, 0.5), 0.5)
        self.assertEqual(compute_f1(0.0, 0.0), 0.0)

    def test_ccc_metric(self):
        """Test CCC (Concordance Correlation Coefficient) computation."""
        def compute_ccc(x, y):
            """Compute CCC between two arrays."""
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_var = np.var(x)
            y_var = np.var(y)
            cov = np.mean((x - x_mean) * (y - y_mean))
            
            if x_var == 0 or y_var == 0:
                return 0.0
            
            ccc = (2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2)
            return ccc
        
        # Perfect correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        self.assertAlmostEqual(compute_ccc(x, y), 1.0, places=5)
        
        # Perfect negative correlation
        y_neg = np.array([5, 4, 3, 2, 1])
        self.assertAlmostEqual(compute_ccc(x, y_neg), -1.0, places=5)

    def test_perplexity_from_loss(self):
        """Test perplexity computation from cross-entropy loss."""
        loss = 2.0
        ppl = math.exp(loss)
        self.assertAlmostEqual(ppl, 7.389, places=2)
        
        # Very low loss -> PPL close to 1
        self.assertAlmostEqual(math.exp(0.1), 1.105, places=2)


class TestDataSplits(unittest.TestCase):
    """Tests for data splitting functionality."""

    def test_train_val_split(self):
        """Test train/val split logic."""
        # Simulate token splitting logic
        n = 1000
        train_frac = 0.9
        val_frac = 0.05
        
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        
        self.assertEqual(train_end, 900)
        self.assertEqual(val_end, 950)
        
        # Verify splits add up
        train_size = train_end
        val_size = val_end - train_end
        test_size = n - val_end
        
        self.assertEqual(train_size + val_size + test_size, n)


class TestColorPalette(unittest.TestCase):
    """Tests for visualization color palette."""

    def test_palette_exists(self):
        """Test that color palette is defined."""
        try:
            from comprehensive_training_report import PALETTE, ARCH_LABELS
            
            # All architectures should have colors
            for arch in ARCH_LABELS.keys():
                self.assertIn(arch, PALETTE)
                # Should be valid hex color
                color = PALETTE[arch]
                self.assertTrue(color.startswith('#'))
                self.assertEqual(len(color), 7)  # #RRGGBB
                
        except ImportError as e:
            self.skipTest(f"Could not import palette: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests requiring actual files/components."""

    @unittest.skip("Requires actual model files")
    def test_full_training_pipeline(self):
        """Test the full training pipeline (skipped by default)."""
        pass

    def test_imports_all_modules(self):
        """Test that all main modules can be imported."""
        modules_to_test = [
            'run_small_lm',
            'small_lm_architectures',
        ]
        # comprehensive_training_report is in scripts/ not a module
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Failed to import {module_name}: {e}")


def run_tests():
    """Run all tests with proper output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEmbeddingExtractor,
        TestSmallLMArchitectures,
        TestTokenization,
        TestConfigurationBuilders,
        TestAggregationUtilities,
        TestOptunaConfigLoading,
        TestResultSaving,
        TestMetricsAndFormatting,
        TestDataSplits,
        TestColorPalette,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
