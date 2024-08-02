import redis
import bittensor as bt

class RedisInterface:
    def __init__(self, host="localhost", port=6379):
        self.host = host
        self.port = port
        self.redis_client = None
        self.is_connected = False

    def connect(self):
        bt.logging.info("Initializing Redis connection")
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.is_connected = True
            bt.logging.info("Redis connection successful")
            return True
        except redis.ConnectionError:
            bt.logging.warning("Failed to connect to Redis server. GUI interfaces will not be available. Only CLI will work.")
            self.is_connected = False
            return False

    def publish(self, channel, message):
        if not self.is_connected:
            bt.logging.warning("Redis is not connected. Cannot publish message.")
            return False
        try:
            self.redis_client.publish(channel, message)
            return True
        except Exception as e:
            bt.logging.error(f"Error publishing to Redis: {e}")
            return False

    def subscribe(self, channel):
        if not self.is_connected:
            bt.logging.warning("Redis is not connected. Cannot subscribe to channel.")
            return None
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            bt.logging.error(f"Error subscribing to Redis channel: {e}")
            return None

    def get(self, key):
        if not self.is_connected:
            bt.logging.warning("Redis is not connected. Cannot get value.")
            return None
        try:
            return self.redis_client.get(key)
        except Exception as e:
            bt.logging.error(f"Error getting value from Redis: {e}")
            return None

    def set(self, key, value):
        if not self.is_connected:
            bt.logging.warning("Redis is not connected. Cannot set value.")
            return False
        try:
            self.redis_client.set(key, value)
            return True
        except Exception as e:
            bt.logging.error(f"Error setting value in Redis: {e}")
            return False