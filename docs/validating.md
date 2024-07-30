# Guide for Validators

1. Obtain API key(s) on the MEGA plan from [this games API](https://rapidapi.com/search/Sports) (currently using API-FOOTBALL and API-BASEBALL). We plan to reduce API calls in the future to make this cheaper, but you will need to periodically add API keys as we add sports (American Football will be needed in mid August).

2. Create a `.env` file in the top-level directory: 

```bash
RAPID_API_KEY=<YOUR_API_KEY>
```

3. Run `source ./scripts/start_neuron.sh` and follow prompts for validator. You can run this script with flags if you prefer not to enter prompts. To use a local subtensor, run `source scripts/start_neuron.sh --subtensor.chain_endpoint <YOUR ENDPOINT>`

>[!NOTE]
> We recommend running with --logging.trace while we are in Beta. This is much more verbose, but it will help us to debug if you run into issues.
