
<div align="center">

# **BetTensor** 
</div>


<details>
<summary>What Is BetTensor?</summary>

BetTensor is a sports prediction subnet. The goal of BetTensor is to provide a platform for sports fans to predict the outcomes of their favorite sporting events, and ML/AI researchers to develop new models and strategies to benchmark against good, old fashioned human intelligence and intuition. Mining on this subnet is simple. In this Beta release, miners receive upcoming games and odds, submit predictions as a simulated wager, and are rewarded for correct predictions upon game conclusion. Compute requirements are minimal for those choosing the human intelligence method - all you need is to consistently submit predictions for upcoming games.



</details>

#
<details>

<summary>Installation and Setup</summary>

To mine or validate on this subnet, we recommend starting with a cheap VPS instance running Ubuntu 22.04 (or 20.04, though this can take longer to set up dependencies). As with most Subnets, we also recommend running your own Lite Node. You can find a guide to running a Lite Node [here](https://docs.bittensor.com/subtensor-nodes/).

Setting up a Lite Node will take care of a lot of the pre-requisites for running a miner or validator. Once you have your Lite Node running, you can clone this repository to your local machine.

```bash
git clone https://github.com/bettensor/bettensor.git
```

We also highly recommend using a virtual environment for setup, either through [venv](https://docs.python.org/3/library/venv.html) or [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/en/latest/).

The virtual environment should be initialized with Python 3.10.

After setting up your virtual environment, install the requirements.

```bash
pip install -r requirements.txt
```

We recommend running validators and miners as pm2 processes, though you can also use tmux if you prefer. To install pm2 and it's dependencies, run:

```bash
sudo apt-get install -y npm jq
sudo npm install -g pm2
```

After completing the initial setup and installation, make sure you have a Bittensor wallet set up. You can find a guide to setting up a Bittensor CLI wallet [here](https://docs.bittensor.com/wallets/).
The hotkey will need to be registered on the subnet to begin mining. 

To register on testnet, you will need test Tao, which you can obtain via a request on the bittensor discord [here](https://discord.com/channels/799672011265015819/1190048018184011867).


Mainnet:
```bash
btcli subnet register --netuid <NETUID> --wallet.name <YOUR_COLDKEY> --wallet.hotkey <YOUR_HOTKEY>
```
BetTensor is {UID} on the BitTensor Mainnet.

Testnet:
```bash
btcli subnet register --netuid <NETUID> --wallet.name <YOUR_COLDKEY> --wallet.hotkey <YOUR_HOTKEY> --subtensor.network test
```
BetTensor is {UID} on the BitTensor Testnet.

After you have reached this point, instructions diverge based on whether you are a validator or miner. 


</details>

#



<details>
<summary>Guide for Validators</summary>

After registering your wallet, validators will need to head over to this [games api](https://bettensor.com/games) to get an api key. 

Once you have your api key, create a .env file in the top level directory and add the following:
```bash
API_KEY=<YOUR_API_KEY>
```
next, open up the validator.sh file in the top level directory. 

You'll see three commands for local, main, and test, with two commented out. Uncomment the one you need (most likely test or main),
and make sure to replace wallet.name and wallet.hotkey with your validator coldkey and hotkey, respectively. 

After you've made these changes and saved, you can run the validator.sh file with pm2 to start the validator:

```bash
pm2 start validator.sh --name validator --interpreter bash
```

check miner logs with:
```bash
pm2 logs validator
```

Note: Because we are in Beta, we will likely encounter bugs and unexpected behavior. Please contact our dev team via the official BitTensor discord channel, or the BetTensor discord, if you run into any issues.




</details>

#

<details>

<summary>Guide for Miners</summary>

After registering your wallet, you are ready to start submitting predictions. 

Open up the miner.sh file in the top level directory. 

You'll see three commands for local, main, and test, with two commented out. Uncomment the one you need (most likely test or main),
and make sure to replace wallet.name and wallet.hotkey with your coldkey and hotkey, respectively. 

After you've made these changes and saved, you can run the miner.sh file with pm2 to start the miner:

```bash
pm2 start miner.sh --name miner --interpreter bash
```

check miner logs with:
```bash
pm2 logs miner
```

fter the miner is started and has received some games from the validators, you can start submitting predictions!

To submit predictions, you will need to run the following command from the top level directory:
```bash
python bettensor/miner/cli.py
```

This will open up a terminal interface that allows you to submit predictions. 

Note: Because we are in Beta, we will likely encounter bugs and unexpected behavior. Please contact our dev team via the official BitTensor discord channel, or the BetTensor discord, if you run into any issues.


</details>

#

<details>

<summary>Details for Current Release Version (v0.0.1, Beta)</summary>


- This is a Beta release of BetTensor. In this current version, we don't have integration for model-based predictions. If you choose to, you are welcome to run your own model by integrating with the miner database, but this is not required.

- Current Supported Sports are: MLB , MLS 

</details>







#

## License
This repository is licensed under the MIT License.
```text
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
```
