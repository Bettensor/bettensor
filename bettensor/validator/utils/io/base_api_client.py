class BaseAPIClient:
    """
    Base class for handling all sports data API requests.
    """

    def __init__(self):
        self.last_update_file = "last_update.json"
        self.last_update_date = self.load_last_update_date()

    def fetch_and_update_game_data(self):
        """
        fetch and update game data from external APIs.
        Override this method in the child classes.
        """

        pass

    def fetch_and_update_single_game_data(self, external_id):
        """
        fetch and update single game data from external APIs.
        Override this method in the child classes.
        """

        pass

    def load_last_update_date(self):
        """
        load the last update date from the last_update.json file.
        """

        pass

    def save_last_update_date(self):
        """
        save the last update date to the last_update.json file.
        """

        pass
