import pickle


class ItemGenerator:

    def __init__(self, file_list: list):
        self.file_list = file_list
        self.file_lengths = [-1 for i in range(len(file_list))]

    def _load_batch(self, cursor: dict, batch_size: int):
        items = []

        while len(items) < batch_size and cursor is not None:
            lines_to_read = batch_size - len(items)
            items.extend(self._load_from_file(cursor, lines_to_read))
            if self._has_more_files(cursor):
                cursor = self._get_next_cursor(cursor, lines_to_read)
            else:
                cursor = None

        return items, cursor

    def _load_from_file(self, cursor, lines_to_read):

        # TODO: make this abstract and move implementation to specific generator: load from csv not from pickle

        with open(self.file_list[cursor["file_index"]], "rb") as file:
            items = pickle.load(file)[cursor["line"]:cursor["line"] + lines_to_read]
        return items

    def _get_item_count(self, file_index: int):

        # TODO: make this abstract and move implementation to specific generator: count items in file for new format

        if self.file_lengths[file_index] == -1:
            with open(self.file_list[file_index], "rb") as file:
                count = len(pickle.load(file))
            self.file_lengths[file_index] = count

        return self.file_lengths[file_index]

    def _get_next_cursor(self, cursor, lines_to_read):
        lines = self._get_item_count(cursor["file_index"])
        if lines > cursor["line"] + lines_to_read:
            return {
                "file_index": cursor["file_index"],
                "line": cursor["line"] + lines_to_read
            }
        else:
            return {
                "file_index": cursor["file_index"] + 1,
                "line": 0
            }

    def _has_more_files(self, cursor: dict):
        return cursor["file_index"] != len(self.file_list) - 1

    def get_item_count(self):
        for index in range(len(self.file_list)):
            if self.file_lengths[index] == -1:
                self._get_item_count(index)
        return sum(self.file_lengths)

    def build_generator(self, batch_size):
        """
        creates a generator which will return one batch of items at the time

        :param batch_size: how many items should be returned at once (default 1)
        :return: item generator
        """
        cursor = {
            "file_index": 0,
            "line": 0
        }

        while cursor is not None:
            batch, cursor = self._load_batch(cursor, batch_size)
            yield batch