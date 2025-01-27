import copy
from pathlib import Path

from immuneML.dsl.instruction_parsers.LabelHelper import LabelHelper
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.util.ParameterValidator import ParameterValidator
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisInstruction import ExploratoryAnalysisInstruction
from immuneML.workflows.instructions.exploratory_analysis.ExploratoryAnalysisUnit import ExploratoryAnalysisUnit


class ExploratoryAnalysisParser:

    """

    The specification consists of a list of analyses that need to be performed;

    Each analysis is defined by a dataset identifier, a report identifier and optionally encoding and labels
    and are loaded into ExploratoryAnalysisUnit objects;

    DSL example for ExploratoryAnalysisInstruction assuming that d1, r1, r2, e1 are defined previously in definitions section:

    .. highlight:: yaml
    .. code-block:: yaml

        instruction_name:
            type: ExploratoryAnalysis
            analyses:
                my_first_analysis:
                    dataset: d1
                    report: r1
                my_second_analysis:
                    dataset: d1
                    encoding: e1
                    report: r2
                    labels:
                        - CD
                        - CMV

    """

    def parse(self, key: str, instruction: dict, symbol_table: SymbolTable, path: Path = None) -> ExploratoryAnalysisInstruction:
        exp_analysis_units = {}

        ParameterValidator.assert_keys(instruction, ["analyses", "type"], "ExploratoryAnalysisParser", "ExploratoryAnalysis")
        for analysis_key, analysis in instruction["analyses"].items():

            params = self._prepare_params(analysis, symbol_table, f"{key}/{analysis_key}")
            exp_analysis_units[analysis_key] = ExploratoryAnalysisUnit(**params)

        process = ExploratoryAnalysisInstruction(exploratory_analysis_units=exp_analysis_units, name=key)
        return process

    def _prepare_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        valid_keys = ["dataset", "report", "preprocessing_sequence", "labels", "encoding", "number_of_processes"]
        ParameterValidator.assert_keys(list(analysis.keys()), valid_keys, "ExploratoryAnalysisParser", "analysis", False)

        params = {"dataset": symbol_table.get(analysis["dataset"]), "report": copy.deepcopy(symbol_table.get(analysis["report"]))}

        optional_params = self._prepare_optional_params(analysis, symbol_table, yaml_location)
        params = {**params, **optional_params}

        return params

    def _prepare_optional_params(self, analysis: dict, symbol_table: SymbolTable, yaml_location: str) -> dict:

        params = {}
        dataset = symbol_table.get(analysis["dataset"])

        if "encoding" in analysis:
            params["encoder"] = symbol_table.get(analysis["encoding"]).build_object(dataset, **symbol_table.get_config(analysis["encoding"])["encoder_params"])

            if "labels" in analysis:
                params["label_config"] = LabelHelper.create_label_config(analysis["labels"], dataset, ExploratoryAnalysisParser.__name__, yaml_location)
            else:
                params["label_config"] = LabelConfiguration()

        if "preprocessing_sequence" in analysis:
            params["preprocessing_sequence"] = symbol_table.get(analysis["preprocessing_sequence"])

        if "number_of_processes" in analysis:
            params["number_of_processes"] = analysis["number_of_processes"]

        return params
