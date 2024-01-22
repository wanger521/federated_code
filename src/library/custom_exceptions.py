
class DataPartitionError(Exception):
    def __init__(self, error_info):
        super().__init__(self)  # Initialize the parent class
        """
        Thrown when there is an empty set in the data partition.
        """
        self.error_info = error_info

    def __str__(self):
        return self.error_info
