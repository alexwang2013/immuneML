from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from immuneML.analysis.data_manipulation.NormalizationType import NormalizationType
from immuneML.data_model.dataset.RepertoireDataset import RepertoireDataset
from immuneML.data_model.repertoire.Repertoire import Repertoire
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.encodings.kmer_frequency.KmerFrequencyEncoder import KmerFrequencyEncoder
from immuneML.encodings.kmer_frequency.ReadsType import ReadsType
from immuneML.encodings.kmer_frequency.sequence_encoding.KmerSequenceEncoder import KmerSequenceEncoder
from immuneML.encodings.kmer_frequency.sequence_encoding.SequenceEncodingType import SequenceEncodingType
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.environment.SequenceType import SequenceType
from immuneML.preprocessing.Preprocessor import Preprocessor
from immuneML.util.PathBuilder import PathBuilder


class PreprocessorLasso(Preprocessor):
    # def __init__(self, lower_limit: int = -1, upper_limit: int = -1, label: str = ' ', k: int = 3):
    def __init__(self, label: str, k: int, alpha: float = 1., criteria: str = 'top'):
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        """
        self.label = label
        self.k = k
        self.alpha = alpha
        self.criteria = criteria

    def process_dataset(self, dataset: RepertoireDataset, result_path: Path = None):
        # Prepare the parameter dictionary for the process method
        params = {'result_path': result_path,
                  'k': self.k,
                  'alpha': self.alpha,
                  'criteria': self.criteria,
                  'label': self.label}
        """"
        if self.lower_limit > -1:
            params["lower_limit"] = self.lower_limit
        if self.upper_limit > -1:
            params["upper_limit"] = self.upper_limit
        """
        return PreprocessorLasso.process(dataset, params)

    @staticmethod
    def process(dataset: RepertoireDataset, params: dict) -> RepertoireDataset:
        path = params['result_path'] / str(params['alpha']) / str(params['k'])
        PathBuilder.build(path)

        dataset3 = dataset.clone()

        encoder = KmerFrequencyEncoder.build_object(dataset, **{
                "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                "reads": ReadsType.UNIQUE.name,
                "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                "sequence_type": SequenceType.AMINO_ACID.name,
                "k": params['k']
            })

        d1 = encoder.encode(dataset, EncoderParams(
            result_path=path / "1/",
            label_config=LabelConfiguration([Label(params['label'])]),  # In the yaml, user specifies label
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        training_input = np.array(d1.encoded_data.examples.A)
        training_output = np.array(d1.encoded_data.labels[params['label']])

        diomio = Lasso(alpha=params['alpha'], fit_intercept=True, normalize=True, max_iter=5000)
        diomio.fit(training_input, training_output) # , sample_weight=sw_train)
        aa = diomio.coef_

        kmer_keep_list = [x for (i, x) in zip(aa, d1.encoded_data.feature_names) if i != 0]
        if not kmer_keep_list:
            return dataset
        logging.info('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')
        logging.info('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')
        logging.info('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')
        print('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')
        print('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')
        print('UUUAHHHAHHAAAAAAAAAAAAAAAAAAAAAA')

        logging.info('Numberical check: population count')
        logging.info(f'{np.sum(training_input) }')
        logging.info(f'{np.sum(training_output) }')
        print('Numberical check: population count')
        print(f'{np.sum(training_input) }')
        print(f'{np.sum(training_output) }')
        # np.set_printoptions(threshold=sys.maxsize)
        # logging.info(f'{training_input}')
        logging.info(f'Coef to keep: {len(kmer_keep_list)}, Ncoefs: {len(aa)}')
        logging.info(' ')
        print(f'Coef to keep: {len(kmer_keep_list)}, Ncoefs: {len(aa)}')
        print(' ')
        aaa = 0
        for i in aa:
            if i == 0:
                aaa += 1
        logging.info(f'N of non-zero coefficients: {aaa}')
        print(f'N of non-zero coefficients: {aaa}')

        def seq_fil(seq, kmer_keep_list):
            """
            Return True if the seq has a kmer contained in the kmer_keep_list
                False otherwise.
            """
            y = KmerSequenceEncoder.encode_sequence(seq, EncoderParams(model={'k': params['k']},
                                                                       label_config=None,
                                                                       result_path=path,
                                                                       pool_size=4))

            if y is not None:
                for i in y:
                    if i in kmer_keep_list:
                        return True
            return False

        dataset3.repertoires = []
        for i, rep in enumerate(d1.repertoires):
            list_seq_filted = list(filter(lambda seq: seq_fil(seq, kmer_keep_list), rep.sequences)) 
            if not list_seq_filted:
                continue
            logging.info(f'SEQ TO KEEP: {len(list_seq_filted)} out of {len(rep.sequences)}')
            print(f'SEQ TO KEEP: {len(list_seq_filted)} out of {len(rep.sequences)}')

            new_repertoire = Repertoire.build_from_sequence_objects(list_seq_filted,
                                                                    path=params['result_path'],
                                                                    metadata=rep.metadata)

            dataset3.repertoires.append(new_repertoire)

        """
        # THE FOLLOWING PART is not needed, but it is to check what is going on later.
        # eg. The reports are identical.
        encoder3 = KmerFrequencyEncoder.build_object(dataset3, **{
                 "normalization_type": NormalizationType.RELATIVE_FREQUENCY.name,
                 "reads": ReadsType.UNIQUE.name,
                 "sequence_encoding": SequenceEncodingType.CONTINUOUS_KMER.name,
                 "sequence_type": SequenceType.AMINO_ACID.name,
                 "k": params['k']})


        d3 = encoder3.encode(dataset3, EncoderParams(
            result_path=path / "3/",
            label_config=LabelConfiguration([Label(params['label'])]),  # In the yaml, user specifies label
            learn_model=True,
            model={},
            filename="dataset.pkl"
        ))

        # PreprocessorLasso.export(aa, d3.encoded_data.feature_names, path)
        logging.info(' ')
        
        logging.info('remaining ')
        logging.info(f'{len(d1.encoded_data.feature_names)}, {len(d3.encoded_data.feature_names)}')
        kmer_keep_list = [x for (i, x) in zip(aa, d1.encoded_data.feature_names) if i != 0]
        logging.info(f'keep list now {len(keep_list)}') 
        """
        if kmer_keep_list:
            return dataset3
        return dataset

    @staticmethod
    def export(lasso_coeff: np.ndarray, features_names: np.ndarray, results: Path):
        dataframe = pd.DataFrame({'features_names': features_names, 'lasso_coeff': lasso_coeff})
        dataframe.to_csv(results / 'filename.csv')


