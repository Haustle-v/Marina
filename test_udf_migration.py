import unittest
import numpy as np
import pyarrow as pa
from typing import List

# Import from the source directory
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from pyseekdb.transformer import udf, UDF
from pyseekdb.udfs import sentence_transformer_udf

class TestUDFMigration(unittest.TestCase):
    def test_simple_scalar_udf(self):
        """Test a simple scalar UDF that multiplies input by 2."""
        
        @udf(input_columns=["a"], data_type=pa.int64())
        def double_val(a: int) -> int:
            return a * 2
            
        # Create a RecordBatch
        data = [
            pa.array([1, 2, 3, 4, 5]),
            pa.array(["a", "b", "c", "d", "e"])
        ]
        batch = pa.RecordBatch.from_arrays(data, names=["a", "b"])
        
        # Execute UDF
        result = double_val(batch)
        
        # Verify result
        self.assertIsInstance(result, pa.Array)
        expected = [2, 4, 6, 8, 10]
        self.assertEqual(result.to_pylist(), expected)
        print("\n✅ Simple scalar UDF test passed")

    def test_udf_validation(self):
        """Test UDF schema validation."""
        
        @udf(input_columns=["non_existent"], data_type=pa.int64())
        def invalid_udf(a: int) -> int:
            return a
            
        schema = pa.schema([("a", pa.int64())])
        
        # Should raise ValueError because input column doesn't exist
        with self.assertRaises(ValueError):
            invalid_udf.validate_against_schema(schema)
        print("\n✅ UDF validation test passed")

    def test_sentence_transformer_udf(self):
        """Test the sentence transformer UDF."""
        print("\nTesting sentence transformer UDF (this may download model)...")
        
        # Use a small model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Create UDF
        # We specify input_columns explicitly or rely on default "text"
        embedding_func = sentence_transformer_udf(
            model=model_name,
            column="text",
            normalize=True
        )
        
        # Create RecordBatch with text
        texts = ["Hello world", "Artificial Intelligence", "OceanBase"]
        batch = pa.RecordBatch.from_arrays(
            [pa.array(texts)], 
            names=["text"]
        )
        
        # Execute UDF
        try:
            embeddings = embedding_func(batch)
            
            # Verify results
            self.assertIsInstance(embeddings, pa.Array)
            self.assertEqual(len(embeddings), 3)
            
            # Check dimension (should be 384 for MiniLM-L6-v2)
            first_embedding = embeddings[0].as_py()
            self.assertEqual(len(first_embedding), 384)
            
            # Check normalization (L2 norm should be close to 1)
            norm = np.linalg.norm(first_embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)
            
            print("✅ Sentence transformer UDF test passed")
            
        except ImportError as e:
            print(f"⚠️ Skipped sentence transformer test: {e}")
        except Exception as e:
            print(f"❌ Sentence transformer test failed: {e}")
            raise e

    def test_udf_type_inference(self):
        """Test UDF return type inference."""
        
        @udf(input_columns=["a"])
        def infer_int(a: int) -> int:
            return a
            
        self.assertEqual(infer_int.data_type, pa.int64())
        
        @udf(input_columns=["a"])
        def infer_float(a: int) -> float:
            return float(a)
            
        self.assertEqual(infer_float.data_type, pa.float32())
        print("\n✅ UDF type inference test passed")

if __name__ == "__main__":
    unittest.main()

