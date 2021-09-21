from sklearn.linear_model import Lasso as SklearnLasso
from immuneML.ml_methods.SklearnMethod import SklearnMethod
from scripts.specification_util import update_docs_per_mapping
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from pathlib import Path


class Lasso(SklearnMethod):
    """
    This is a wrapper of scikit-learn’s Lasso class. Please see the
    `scikit-learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_
    of Lasso for the parameters.

    Note: if you are interested in plotting the coefficients of the logistic regression model,
    consider running the :ref:`Coefficients` report.

    For usage instructions, check :py:obj:`~immuneML.ml_methods.SklearnMethod.SklearnMethod`.


    YAML specification:

    .. indent with spaces
    .. code-block:: yaml

        my_logistic_regression: # user-defined method name
            Lasso: # name of the ML method
                # sklearn parameters (same names as in original sklearn class)
                penalty: l1 # always use penalty l1
                C: [0.01, 0.1, 1, 10, 100] # find the optimal value for C
                # Additional parameter that determines whether to print convergence warnings
                show_warnings: True
            # if any of the parameters under Lasso is a list and model_selection_cv is True,
            # a grid search will be done over the given parameters, using the number of folds specified in model_selection_n_folds,
            # and the optimal model will be selected
            model_selection_cv: True
            model_selection_n_folds: 5
        # alternative way to define ML method with default values:
        my_default_logistic_regression: Lasso

    """
    default_parameters = {'alpha': 1., 'fit_intercept': True, 'normalize': False, 
                          'precompute': False, 'max_iter': 1000, 'tol': 0.0001,
                          'selection': 'cyclic'}

    def __init__(self, parameter_grid: dict = None, parameters: dict = None):
        parameters = {**self.default_parameters, **(parameters if parameters is not None else {})}

        if parameter_grid is not None:
            parameter_grid = parameter_grid
        else:
            parameter_grid = {'alpha': [1.]}

        super(Lasso, self).__init__(parameter_grid=parameter_grid, parameters=parameters)

    def _get_ml_model(self, cores_for_training: int = 2, X=None):
        params = self._parameters.copy()
        # params["n_jobs"] = cores_for_training
        return SklearnLasso(**params)

    def can_predict_proba(self) -> bool:
        return False

    def get_params(self):
        params = self.model.get_params()
        params["coefficients"] = self.model.coef_[0].tolist()
        params["intercept"] = self.model.intercept_.tolist()
        return params

    def predict(self, encoded_data: EncodedData, label_name: str):
        return {label_name: self.model.predict(encoded_data.examples)}

    def store(self, path: Path, feature_names=None, details_path: Path = None):
        return
