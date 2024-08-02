import hashlib
import json
from bettensor.miner.utils.serialization import custom_serializer

class CacheManager:
    def __init__(self):
        self.game_hashes = {}
        self.cached_predictions = {}

    def _hash_game(self, game):
        try:
            game_json = json.dumps(game, sort_keys=True, default=custom_serializer)
        except TypeError:
            # If JSON serialization fails, use a string representation of the game
            game_json = str(game)
        return hashlib.md5(game_json.encode()).hexdigest()

    def filter_changed_games(self, games):
        changed_games = {}
        for game_id, game_data in games.items():
            game_hash = self._hash_game(game_data)
            if game_id not in self.game_hashes or self.game_hashes[game_id] != game_hash:
                changed_games[game_id] = game_data
                self.game_hashes[game_id] = game_hash
        return changed_games

    def update_cached_predictions(self, predictions):
        self.cached_predictions = predictions

    def get_cached_predictions(self):
        return self.cached_predictions