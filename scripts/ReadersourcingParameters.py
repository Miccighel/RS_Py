class ReadersourcingParameters:

    def __init__(
            self,
            dataset_name="ground_truth_1",
            dataset_folder_path="../data/{}/".format("ground_truth_1"),
            days_serialization=False,
            days_serialization_threshold=0,
            days_number=0,
            current_day=0,
            days_serialization_cleaning=False,
            days_cleaning_threshold=0,
            data_shuffled=False,
            shuffle_amount=0,
            current_shuffle=0,
            result_compression=False,
            archive_name="archive"
    ):

        self.dataset_name = dataset_name
        self.dataset_folder_path = dataset_folder_path.format(dataset_name)

        self.days_serialization = days_serialization
        self.days_serialization_threshold = days_serialization_threshold
        self.days_number = days_number
        self.current_day = current_day
        self.days_serialization_cleaning = days_serialization_cleaning
        self.days_cleaning_threshold = days_cleaning_threshold
        self.data_shuffled = data_shuffled
        self.shuffle_amount = shuffle_amount
        self.current_shuffle = current_shuffle
        self.result_compression = result_compression
        self.archive_name = archive_name

        self.result_folder_base_path = "../models/{}/readersourcing/".format(dataset_name)
        paths = []
        for shuffle_index in range(shuffle_amount):
            paths.append("{}shuffle_{}/".format(self.result_folder_base_path, shuffle_index))
        self.result_shuffle_paths = paths

