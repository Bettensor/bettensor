# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import copy
import typing

import bittensor as bt
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from os import path, makedirs
# Sync calls set weights and also resyncs the metagraph.
from bettensor.utils.config import check_config, add_args, config
from bettensor.utils.misc import ttl_get_block
from bettensor import __spec_version__ as spec_version
from bettensor.mock import MockSubtensor, MockMetagraph


class BaseNeuron():
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """
    def __init__(self, parser: ArgumentParser, profile: str, config=None):
        self.parser = parser
        self.profile = profile
        self.step = 0
        self.last_updated_block = 0
        self.base_path = f"{path.expanduser('~')}/bettensor" 
    def config(self, bt_classes: list) -> bt.config:
        """Applies neuron configuration.

        This function attaches the configuration parameters to the
        necessary bittensor classes and initializes the logging for the
        neuron.

        Args:
            bt_classes:
                A list of Bittensor classes the apply the configuration
                to

        Returns:
            config:
                An instance of Bittensor config class containing the
                neuron configuration

        Raises:
            AttributeError:
                An error occurred during the configuration process
            OSError:
                Unable to create a log path.

        """
        try:
            for bt_class in bt_classes:
                bt_class.add_args(self.parser)
        except AttributeError as e:
            bt.logging.error(
                f"Unable to attach ArgumentParsers to Bittensor classes: {e}"
            )
            raise AttributeError from e

        config = bt.config(self.parser)

        # Construct log path
        log_path = f"{self.base_path}/logs/{config.wallet.name}/{config.wallet.hotkey}/{config.netuid}/{self.profile}"

        # Create the log path if it does not exists
        try:
            config.full_path = path.expanduser(log_path)
            if not path.exists(config.full_path):
                makedirs(config.full_path, exist_ok=True)
        except OSError as e:
            bt.logging.error(f"Unable to create log path: {e}")
            raise OSError from e

        return config
