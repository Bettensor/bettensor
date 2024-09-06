# Guide for Validators

## Option 1: Bettensors Central API

1. This API distributes games from RapidAPI and Bet365 so that you don't have to pay for an api key. Just select the option of "use_bt_api" in the setup script, or pass it manually when running validator.py.

## Option 2: RapidAPI and Bet365 for Odds

1. Obtain API key(s) on the MEGA plan from [this games API](https://rapidapi.com/search/Sports) (currently using API-FOOTBALL and API-BASEBALL). 

2. Obtain an API key from betsapi.com. Select the EVERYTHING API.

3. Create a `.env` file in the top-level directory: 

```bash
RAPID_API_KEY=<YOUR_API_KEY>
BET365_API_KEY=<YOUR_API_KEY>
```

4. Run `source ./scripts/start_neuron.sh` and follow prompts for validator. You can run this script with flags if you prefer not to enter prompts. To use a local subtensor, run `source scripts/start_neuron.sh --subtensor.chain_endpoint <YOUR ENDPOINT>`

>[!NOTE]
> We recommend running with --logging.trace while we are in Beta. This is much more verbose, but it will help us to debug if you run into issues.






