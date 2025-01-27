import os
import shutil
from unittest import TestCase

import numpy as np
from pathlib import Path

from immuneML.caching.CacheType import CacheType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.filtered_sequence_encoding.CompAIRRSequenceAbundanceEncoder import CompAIRRSequenceAbundanceEncoder
from immuneML.environment.Constants import Constants
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.RepertoireBuilder import RepertoireBuilder


class TestCompAIRRSequenceAbundanceEncoder(TestCase):
    def setUp(self) -> None:
        os.environ[Constants.CACHE_TYPE] = CacheType.TEST.name

    def test_encode(self):
        compairr_paths = [Path("/usr/local/bin/compairr"), Path("./compairr/src/compairr")]

        for compairr_path in compairr_paths:
            if compairr_path.exists():
                self._test_encode(compairr_path)
                break

    def _build_test_dataset(self, path):
        repertoires, metadata = RepertoireBuilder.build([["GGG", "III", "LLL", "MMM"],
                                                         ["DDD", "EEE", "FFF", "III", "LLL", "MMM"],
                                                         ["CCC", "FFF", "MMM"],
                                                         ["AAA", "CCC", "EEE", "FFF", "LLL", "MMM"]],
                                                        labels={"l1": [True, True, False, False]}, path=path)

        dataset = RepertoireDataset(repertoires=repertoires, metadata_file=metadata, identifier="1")

        return dataset

    def _test_encode(self, compairr_path):
        path = EnvironmentSettings.tmp_test_path / "compairr_abundance_encoder/"
        PathBuilder.build(path)

        dataset = self._build_test_dataset(path)

        for ignore_genes in [True, False]:
            result_path = path / f"ignore_genes={ignore_genes}"

            encoder = CompAIRRSequenceAbundanceEncoder.build_object(dataset, **{
                "p_value_threshold": 0.4, "compairr_path": compairr_path, "sequence_batch_size": 2, "ignore_genes": ignore_genes, "threads": 8
            })

            label_config = LabelConfiguration([Label("l1", [True, False], positive_class=True)])

            encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=result_path, label_config=label_config))

            self.assertTrue(np.array_equal(np.array([[1, 4], [1, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

            encoder.p_value_threshold = 0.05

            encoded_dataset = encoder.encode(dataset, EncoderParams(result_path=result_path, label_config=label_config))

            self.assertTrue(np.array_equal(np.array([[0, 4], [0, 6], [0, 3], [0, 6]]), encoded_dataset.encoded_data.examples))

        shutil.rmtree(path)

    def test_find_sequence_p_values_with_fisher(self):
        encoder = CompAIRRSequenceAbundanceEncoder(p_value_threshold=None,
                                                   compairr_path="",
                                                   sequence_batch_size=1000,
                                                   ignore_genes=None,
                                                   threads=None)

        sequence_presence_matrix = np.array([[1, 0, 0, 0],
                                             [1, 0, 1, 1],
                                             [1, 1, 1, 1],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0]])

        is_first_class = np.array([True, False, False, True])

        pvals = encoder._find_sequence_p_values_with_fisher(sequence_presence_matrix, is_first_class)

        expected = [2, 0.5, 1.0, 2, 2]

        for real, expected in zip(pvals, expected):
            self.assertAlmostEqual(real, expected, places=7)


    def test_build_abundance_matrix(self):
        encoder = CompAIRRSequenceAbundanceEncoder(p_value_threshold=None,
                                                   compairr_path="",
                                                   sequence_batch_size=1000,
                                                   ignore_genes=None,
                                                   threads=None)

        sequence_presence_matrix = np.array([[0, 1], [1, 1], [1, 0]])
        matrix_repertoire_ids = np.array(["r1", "r2"])
        dataset_repertoire_ids = ["r2"]
        sequence_p_values_indices = [True, False, True]

        abundance_matrix = encoder._build_abundance_matrix(sequence_presence_matrix, matrix_repertoire_ids,
                                                           dataset_repertoire_ids, sequence_p_values_indices)

        expected = np.array([[1., 2.]])

        self.assertTrue((abundance_matrix == expected).all())

