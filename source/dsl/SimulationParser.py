from source.dsl.SymbolTable import SymbolTable
from source.dsl.SymbolType import SymbolType
from source.simulation.Simulation import Simulation
from source.simulation.implants.Motif import Motif
from source.simulation.implants.Signal import Signal
from source.simulation.motif_instantiation_strategy.MotifInstantiationStrategy import MotifInstantiationStrategy
from source.simulation.signal_implanting_strategy.HealthySequenceImplanting import HealthySequenceImplanting
from source.simulation.signal_implanting_strategy.SignalImplantingStrategy import SignalImplantingStrategy
from source.simulation.signal_implanting_strategy.sequence_implanting.GappedMotifImplanting import GappedMotifImplanting
from source.util.ReflectionHandler import ReflectionHandler


class SimulationParser:
    """
    Simulation should be defined in the following manner:

    .. highlight:: yaml
    .. code-block:: yaml

        simulation:
            motifs:
                m1:
                    seed: AAC
                    instantiation: GappedKmer
                    # probability that when hamming distance is allowed a letter in the seed will be replaced by
                    # other alphabet letters - alphabet_weights
                    alphabet_weights:
                        A: 0.2
                        C: 0.2
                        D: 0.4
                        E: 0.2
                    # Relative probabilities of choosing each position for hamming distance modification.
                    # The probabilities will be scaled to sum to one - position_weights
                    position_weights:
                        0: 1
                        1: 0
                        2: 0
                    params:
                        hamming_distance_probabilities:
                            0: 0.5 # Hamming distance of 0 (no change) with probability 0.5
                            1: 0.5 # Hamming distance of 1 (one letter change) with probability 0.5
                        min_gap: 0
                        max_gap: 1

            signals:
                s1:
                    motifs:
                        - m1
                    sequence_position_weights: # likelihood of implanting at IMGT position of receptor sequence
                        107: 0.5
                    implanting: HealthySequences
            implanting:
                i1:
                    dataset_implanting_rate: 0.5 # percentage of repertoire where the signals will be implanted
                    repertoire_implanting_rate: 0.01 # percentage of sequences within repertoire where the signals will be implanted
                    signals:
                        - s1

    """

    @staticmethod
    def parse_simulation(workflow_specification: dict, symbol_table: SymbolTable):
        if "simulation" in workflow_specification.keys():
            simulation = workflow_specification["simulation"]
            assert "motifs" in simulation, "Workflow specification parser: no motifs were defined for the simulation."
            assert "signals" in simulation, "Workflow specification parser: no signals were defined for the simulation."

            symbol_table = SimulationParser._extract_motifs(simulation, symbol_table)
            symbol_table = SimulationParser._extract_signals(simulation, symbol_table)
            symbol_table = SimulationParser._add_signals_to_implanting(simulation, symbol_table)

        return symbol_table, workflow_specification["simulation"] if "simulation" in workflow_specification else {}

    @staticmethod
    def _add_signals_to_implanting(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        assert sum([settings["dataset_implanting_rate"] for settings in simulation["implanting"].values()]) <= 1, \
            "The total dataset implanting rate can not exceed 1"

        for key in simulation["implanting"].keys():

            item = Simulation(
                dataset_implanting_rate=simulation["implanting"][key]["dataset_implanting_rate"],
                repertoire_implanting_rate=simulation["implanting"][key]["repertoire_implanting_rate"],
                signals=[signal.item for signal in symbol_table.get_by_type(SymbolType.SIGNAL)
                         if signal.item.id in simulation["implanting"][key]["signals"]],
                name=key
            )

            symbol_table.add(key, SymbolType.SIMULATION, item)

        return symbol_table

    @staticmethod
    def _extract_motifs(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        for key in simulation["motifs"].keys():
            instantiation_strategy = SimulationParser._get_instantiation_strategy(simulation["motifs"][key])

            motif = Motif(key, instantiation_strategy,
                          seed=simulation["motifs"][key]["seed"],
                          alphabet_weights=simulation["motifs"][key]["alphabet_weights"],
                          position_weights=simulation["motifs"][key]["position_weights"])
            symbol_table.add(key, SymbolType.MOTIF, motif)
        return symbol_table

    @staticmethod
    def _extract_signals(simulation: dict, symbol_table: SymbolTable) -> SymbolTable:
        for key in simulation["signals"].keys():
            implanting_strategy = SimulationParser._get_implanting_strategy(simulation["signals"][key])
            signal_motifs = [symbol_table.get(motif_id) for motif_id in simulation["signals"][key]["motifs"]]
            signal = Signal(key, signal_motifs, implanting_strategy)
            symbol_table.add(key, SymbolType.SIGNAL, signal)
        return symbol_table

    @staticmethod
    def _get_implanting_strategy(signal: dict) -> SignalImplantingStrategy:
        if "implanting" in signal and signal["implanting"] == "HealthySequences":
            implanting_strategy = HealthySequenceImplanting(GappedMotifImplanting(),
                                                            signal["sequence_position_weights"] if
                                                            "sequence_position_weights" in signal else None)
        else:
            raise NotImplementedError
        return implanting_strategy

    @staticmethod
    def _get_instantiation_strategy(motif_item: dict) -> MotifInstantiationStrategy:
        if "instantiation" in motif_item:
            params = motif_item["params"] if "params" in motif_item else {}
            return ReflectionHandler.get_class_by_name("{}Instantiation".format(motif_item["instantiation"]))(**params)
