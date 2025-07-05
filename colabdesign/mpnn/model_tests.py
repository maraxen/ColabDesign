import os
import tempfile
import unittest

import mdtraj as md
import numpy as np
import pandas as pd
from model import mk_mpnn_model


class TestInputConsistency(unittest.TestCase):
    def setUp(self):
        """Set up a mock mdtraj trajectory and the model instance for each test."""
        print(f"\n--- Running test: {self._testMethodName} ---")
        atom_data = {
            "name": ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O"],
            "element": ["N", "C", "C", "O", "C", "N", "C", "C", "O"],
            "resName": ["ALA", "ALA", "ALA", "ALA", "ALA", "GLY", "GLY", "GLY", "GLY"],
            "resSeq": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "chainID": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "formal_charge": [0] * 9,
        }
        atom_data["serial"] = np.arange(1, len(atom_data["name"]) + 1)
        atoms = pd.DataFrame(atom_data)
        xyz = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.2, 0.1, 0.0],
                    [0.2, 0.2, 0.0],
                    [0.1, -0.1, 0.1],
                    [0.3, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.5, 0.1, 0.0],
                    [0.5, 0.2, 0.0],
                ]
            ]
        )
        self.mdtraj_frame = md.Trajectory(xyz, md.Topology.from_dataframe(atoms))
        assert self.mdtraj_frame is not None, "MDTraj frame should not be None"
        assert self.mdtraj_frame.topology is not None, "MDTraj topology should not be None"
        self.mdtraj_frame.topology.chain(0).id = "A"

        try:
            self.model = mk_mpnn_model(model_name="v_48_020")
        except Exception:
            print("Warning: Could not load model weights. Using MockModel for input prep testing.")

            class MockModel(mk_mpnn_model):
                def __init__(self):
                    self._inputs = {}
                    self._tied_lengths = False

            self.model = MockModel()

    def test_input_consistency(self):
        """
        Validates that all three input methods produce identical numerical inputs.
        """
        pdb_string_content = ""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb") as tmp:
            pdb_filepath = tmp.name
            self.mdtraj_frame.save_pdb(pdb_filepath)
            tmp.seek(0)
            pdb_string_content = tmp.read()

        print("Method 1: Loading from PDB file...")
        self.model.prep_inputs(pdb_filename=pdb_filepath, chain="A")
        inputs_from_file = self.model._inputs

        os.remove(pdb_filepath)

        print("Method 2: Loading from PDB string buffer...")
        self.model.prep_inputs(pdb_string=pdb_string_content, chain="A")
        inputs_from_buffer = self.model._inputs

        print("Method 3: Loading directly from mdtraj frame...")
        self.model.prep_inputs(mdtraj_frame=self.mdtraj_frame, chain="A")
        inputs_from_mdtraj = self.model._inputs

        print("Validating results...")

        self.assert_inputs_equal(inputs_from_file, inputs_from_buffer, "File vs. Buffer")

        self.assert_inputs_equal(inputs_from_file, inputs_from_mdtraj, "File vs. Direct mdtraj")

    def assert_inputs_equal(self, inputs1, inputs2, comparison_name):
        """Helper function to compare two '_inputs' dictionaries."""
        print(f"  - Comparing: {comparison_name}")

        keys_to_check = ["X", "mask", "S", "residue_idx", "chain_idx", "lengths"]

        for key in keys_to_check:
            with self.subTest(f"{comparison_name} - key: {key}"):
                self.assertIn(key, inputs1, f"Key '{key}' not in first input dict.")
                self.assertIn(key, inputs2, f"Key '{key}' not in second input dict.")

                val1 = inputs1[key]
                val2 = inputs2[key]

                if key == "X":  # Floating point coordinates
                    np.testing.assert_allclose(
                        val1,
                        val2,
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=f"Mismatch in '{key}' for {comparison_name}",
                    )
                else:
                    np.testing.assert_array_equal(
                        val1, val2, err_msg=f"Mismatch in '{key}' for {comparison_name}"
                    )
        print(f"  - OK: {comparison_name} inputs are consistent.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
