import functools
import math
import traceback
import numpy as np
import torch
import sqlite3
from datetime import datetime, timezone, timedelta
import asyncio
import bittensor as bt
from bettensor import __spec_version__


class WeightSetter:
    def __init__(
        self,
        metagraph,
        wallet,
        subtensor,
        neuron_config,
        db_path,
    ):
        self.metagraph = metagraph
        self.wallet = wallet
        self.subtensor = subtensor
        self.neuron_config = neuron_config

        self.db_path = db_path

    def connect_db(self):
        return sqlite3.connect(self.db_path)

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    async def set_weights(self, weights: torch.Tensor):
        np.set_printoptions(precision=8, suppress=True)

        bt.logging.info(f"Normalized weights: {weights}")

        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        stake = float(self.metagraph.S[uid])
        if stake < 1000.0:
            bt.logging.error("Insufficient stake. Failed in setting weights.")
            return False

        NUM_RETRIES = 3
        for i in range(NUM_RETRIES):
            bt.logging.info(
                f"Attempting to set weights, attempt {i+1} of {NUM_RETRIES}"
            )
            try:
                loop = asyncio.get_running_loop()
                func = functools.partial(
                    self.subtensor.set_weights, 
                    netuid=self.neuron_config.netuid, 
                    wallet=self.wallet, 
                    uids=self.metagraph.uids, 
                    weights=weights, 
                    version_key=__spec_version__, 
                    wait_for_inclusion=False, 
                    wait_for_finalization=True, 
                )
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func),
                    timeout=90,
                )
            
                bt.logging.trace(f"Set weights result: {result}")

                if isinstance(result, tuple) and len(result) >= 1:
                    success = result[0]
                    bt.logging.trace(f"Set weights message: {success}")
                    if success:
                        bt.logging.info("Successfully set weights.")
                        return True
                    
                else:
                    bt.logging.warning(
                        f"Unexpected result format in setting weights: {result}"
                    )
            except TimeoutError:
                bt.logging.error("Timeout occurred while setting weights.")
            except Exception as e:
                bt.logging.error(f"Error setting weights: {str(e)}")
                bt.logging.error(f"Error traceback: {traceback.format_exc()}")

            if i < NUM_RETRIES - 1:
                await asyncio.sleep(10)

        bt.logging.error("Failed to set weights after all attempts.")
        return False
