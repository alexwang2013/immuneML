import os
from multiprocessing.pool import Pool
from shutil import copyfile

import numpy as np
import pandas as pd

from source.caching.CacheHandler import CacheHandler
from source.data_model.dataset.RepertoireDataset import RepertoireDataset
from source.logging.Logger import log
from source.pairwise_repertoire_comparison.ComparisonData import ComparisonData
from source.util.PathBuilder import PathBuilder

global_comp_data = None
comp_fn = None


class PairwiseRepertoireComparison:

    @log
    def __init__(self, matching_columns: list, item_columns: list, path: str, batch_size: int, pool_size: int):
        self.matching_columns = matching_columns
        self.item_columns = item_columns
        self.path = path
        PathBuilder.build(path)
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.comparison_data = None
        self.comparison_fn = None

    @log
    def create_comparison_data(self, dataset: RepertoireDataset) -> ComparisonData:

        comparison_data = ComparisonData(dataset.get_repertoire_ids(), self.matching_columns, self.pool_size,
                                         self.batch_size, self.path)
        comparison_data.process_dataset(dataset)
        comparison_data = self.add_files_to_cache(comparison_data, dataset)

        return comparison_data

    def add_files_to_cache(self, comparison_data: ComparisonData, dataset: RepertoireDataset) -> ComparisonData:

        cache_paths = []

        for index, batch_path in enumerate(comparison_data.batch_paths):
            cache_paths.append(CacheHandler.get_file_path() + "dataset_{}_batch_{}.csv".format(dataset.identifier, index))
            copyfile(batch_path, cache_paths[-1])

        comparison_data.batch_paths = cache_paths

        return comparison_data

    def prepare_caching_params(self, dataset: RepertoireDataset):
        return (
            ("dataset_identifier", dataset.identifier),
            ("item_attributes", self.item_columns)
        )

    def compare(self, dataset: RepertoireDataset, comparison_fn, comparison_fn_name):
        return CacheHandler.memo_by_params((("dataset_identifier", dataset.identifier),
                                            "pairwise_comparison",
                                            ("comparison_fn", comparison_fn_name)),
                                           lambda: self.compare_repertoires(dataset, comparison_fn))

    def memo_by_params(self, dataset: RepertoireDataset):
        # TODO: refactor this to be immune to removing the cache halfway through repertoire comparison
        comparison_data = CacheHandler.memo_by_params(self.prepare_caching_params(dataset), lambda: self.create_comparison_data(dataset))
        if all(os.path.isfile(path) for path in comparison_data.batch_paths):
            return comparison_data
        else:
            return self.create_comparison_data(dataset)

    @log
    def compare_repertoires(self, dataset: RepertoireDataset, comparison_fn):
        self.comparison_data = self.memo_by_params(dataset)
        repertoire_count = dataset.get_example_count()
        comparison_result = np.zeros([repertoire_count, repertoire_count])
        repertoire_identifiers = dataset.get_repertoire_ids()

        global global_comp_data
        global_comp_data = self.comparison_data
        global comp_fn
        comp_fn = comparison_fn

        arguments = self.prepare_paralellization_arguments(repertoire_count, repertoire_identifiers, comparison_result)

        with Pool(self.pool_size) as pool:
            output = pool.starmap(PairwiseRepertoireComparison.helper_fn, arguments, chunksize=int(len(arguments)/self.pool_size))

        del global_comp_data
        del comp_fn

        counter = 0
        for index1 in range(repertoire_count):
            for index2 in range(index1+1, repertoire_count):
                comparison_result[index1, index2] = output[counter]
                comparison_result[index2, index1] = comparison_result[index1, index2]
                counter += 1

        comparison_df = pd.DataFrame(comparison_result, columns=repertoire_identifiers, index=repertoire_identifiers)

        return comparison_df

    def prepare_paralellization_arguments(self, repertoire_count: int, repertoire_identifiers: list, comparison_result):

        arguments = []

        for index1 in range(repertoire_count):
            comparison_result[index1, index1] = 1
            rep1 = repertoire_identifiers[index1]
            for index2 in range(index1+1, repertoire_count):
                rep2 = repertoire_identifiers[index2]
                arguments.append((rep1, rep2))

        return arguments

    @staticmethod
    def helper_fn(rep_id1: str, rep_id2: str):
        print("Comparing repertoires: {} and {}".format(rep_id1, rep_id2))
        rep1 = global_comp_data.get_repertoire_vector(rep_id1)
        rep2 = global_comp_data.get_repertoire_vector(rep_id2)
        res = comp_fn(rep1, rep2)
        del rep1
        del rep2
        return res
