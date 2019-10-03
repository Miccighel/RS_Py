import pandas as pd


class ReadersourcingToolkit:

    # ------------------------------
    # -------- INIT METHOD ---------
    # ------------------------------

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
            shuffle_special=False,
            result_compression=False,
            archive_name="archive"
    ):

        self.dataset_name = dataset_name
        self.dataset_folder_path = dataset_folder_path.format(dataset_name)
        self.dataset_entries_path = "{}/entries/".format(self.dataset_folder_path)
        self.dataset_shuffle_path = "{}/shuffle/".format(self.dataset_folder_path)

        self.days_serialization = days_serialization
        self.days_serialization_threshold = days_serialization_threshold
        self.days_number = days_number
        self.current_day = current_day
        self.days_serialization_cleaning = days_serialization_cleaning
        self.days_cleaning_threshold = days_cleaning_threshold
        self.data_shuffled = data_shuffled
        self.shuffle_amount = shuffle_amount
        self.current_shuffle = current_shuffle
        self.shuffle_special = shuffle_special
        if self.shuffle_special:
            self.dataset_shuffle_path = "{}/shuffle_special/".format(self.dataset_folder_path)
        self.result_compression = result_compression
        self.archive_name = archive_name

        self.info_filename = "{}info.csv".format(self.dataset_entries_path)
        self.ratings_filename = "{}ratings.csv".format(self.dataset_entries_path)
        self.authors_filename = "{}authors.csv".format(self.dataset_entries_path)
        self.shuffle_filename = "shuffle_{}.csv".format(self.current_shuffle)
        if self.shuffle_special:
            self.info_filename = "{}info_special.csv".format(self.dataset_entries_path)
        self.result_folder_base_path = "../models/{}/readersourcing/".format(dataset_name)
        self.result_quantities_filename = "quantities.json"
        self.result_ratings_filename = "ratings.csv"
        self.result_goodness_filename = "goodness.csv"
        self.result_info_filename = "info.json"
        self.result_days_base_path = "{}days/".format(self.result_folder_base_path)
        self.result_day_folder = "day_{}/".format(self.current_day)
        if self.shuffle_special:
            self.result_shuffle_base_path = "{}shuffle_special/".format(self.result_folder_base_path)
        else:
            self.result_shuffle_base_path = "{}shuffle/".format(self.result_folder_base_path)
        self.result_shuffle_folder = "shuffle_{}/".format(self.current_shuffle)
        paths = []
        for shuffle_index in range(shuffle_amount):
            paths.append("{}shuffle_{}/".format(self.result_shuffle_base_path, shuffle_index))
        self.result_shuffle_paths = paths

    # ----------------------------------------------
    # ---------- OUTPUT HANDLING METHODS  ----------
    # ----------------------------------------------

    def update_day(self):
        self.result_day_folder = "day_{}/".format(self.current_day)

    def update_shuffle(self, current_shuffle):
        self.current_shuffle = current_shuffle
        self.result_shuffle_folder = "shuffle_{}/".format(self.current_shuffle)

    # Returns a dataframe with one column
    # First column: identifiers of the chosen entity
    def extract_identifiers(self, entity):
        identifiers_df = pd.DataFrame(columns=["Identifier"])
        quantities = pd.read_json("{}quantities.json".format(self.result_folder_base_path))
        if entity == "Paper":
            identifiers = quantities.at[1, "Identifiers"]
        else:
            identifiers = quantities.at[3, "Identifiers"]
        for index, identifier in enumerate(identifiers):
            identifiers_df = identifiers_df.append({"Identifier": identifier}, ignore_index=True)
            identifiers_df["Identifier"] = identifiers_df["Identifier"].astype(int)
        return identifiers_df

    # Returns a dataframe with one column
    # First column: identifiers of the chosen entity
    def extract_ratings_df(self):
        papers_ids = self.extract_identifiers("Paper")
        readers_ids = self.extract_identifiers("Reader")
        ratings_df = pd.read_csv("{}ratings.csv".format(self.result_folder_base_path), header=None)
        ratings_df.columns = papers_ids["Identifier"].values
        ratings_df.set_index(readers_ids["Identifier"].values)
        return ratings_df

    # Returns a dataframe with two columns
    # First column: identifiers for the chosen Readersourcing quantity (i.e., if the quantity is "Paper Score" then the
    #                column will contain the id of each paper
    # Second column: values of the parsed quantity
    def build_quantity_df(self, quantity_label, identifiers_perc):
        quantity_df = pd.DataFrame(columns=["Identifier", "Quantity"])
        quantities = pd.read_json("{}quantities.json".format(self.result_folder_base_path))
        row = quantities.loc[quantities["Quantity"] == quantity_label]
        row = row.reset_index()
        identifiers = row.at[0, "Identifiers"]
        if identifiers_perc != 0:
            identifiers_amount = round((len(identifiers) * identifiers_perc) / 100)
            identifiers = identifiers[:identifiers_amount]
        quantity = row.at[0, "Values"]
        for index, identifier in enumerate(identifiers):
            quantity_df = quantity_df.append({"Identifier": identifier, "Quantity": quantity[index]}, ignore_index=True)
        quantity_df["Identifier"] = quantity_df["Identifier"].astype(int)
        return quantity_df

    # Returns a dataframe with three columns
    # First column: indexes of the parsed shuffles
    # Second column: identifiers for the chosen Readersourcing quantity (i.e., if the quantity is "Paper Score" then the
    #                column will contain the id of each paper
    # Third column: values of the parsed quantity
    def build_quantity_df_shuffle(self, quantity_label="Paper Score", shuffle_perc=20, identifiers_perc=20, dump_zeros=False):
        quantity_df = pd.DataFrame(columns=["Shuffle", "Identifier", "Quantity"])
        shuffle_amount = round((self.shuffle_amount * shuffle_perc) / 100)
        for index_shuffle in range(shuffle_amount):
            percentage = 100 * index_shuffle / self.shuffle_amount
            if percentage % 5 == 0:
                print("{}/{} ({}/100%)".format(int(index_shuffle), self.shuffle_amount, int(percentage)))
            path = "{}shuffle_{}/quantities.json".format(self.result_shuffle_base_path, index_shuffle)
            quantities = pd.read_json(path)
            row = quantities.loc[quantities['Quantity'] == quantity_label]
            row = row.reset_index()
            identifiers = row.at[0, "Identifiers"]
            if identifiers_perc != 0:
                identifiers_amount = round((len(identifiers) * identifiers_perc) / 100)
                identifiers = identifiers[:identifiers_amount]
            quantity = row.at[0, "Values"]
            for index, identifier in enumerate(identifiers):
                if dump_zeros:
                    if quantity[index] > 0:
                        quantity_df = quantity_df.append(
                            {"Shuffle": index_shuffle, "Identifier": identifier, "Quantity": quantity[index]},
                            ignore_index=True)
                else:
                    quantity_df = quantity_df.append(
                        {"Shuffle": index_shuffle, "Identifier": identifier, "Quantity": quantity[index]},
                        ignore_index=True)
        quantity_df["Shuffle"] = quantity_df["Shuffle"].astype(int)
        quantity_df["Identifier"] = quantity_df["Identifier"].astype(int)
        print("{}/{} (100/100%)".format(self.shuffle_amount, self.shuffle_amount))
        return quantity_df
