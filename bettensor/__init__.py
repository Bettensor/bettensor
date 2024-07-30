import configparser
from os import path, environ
from dotenv import load_dotenv
import bittensor as bt

load_dotenv()

# Read version
config = configparser.ConfigParser()
setup_file = path.dirname(path.dirname(path.abspath(__file__))) + "/setup.cfg"

config.read(setup_file)

__version__ = config["metadata"]["version"]
__database_version__ = config["metadata"]["database_version"]
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
