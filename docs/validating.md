# Guide for Validators

1. Obtain API key(s) from [this games API](https://rapidapi.com/search/Sports) (currently using API-FOOTBALL and API-BASEBALL).

2. Create a `.env` file in the top-level directory: 

```bash
RAPID_API_KEY=<YOUR_API_KEY>
```

3. Run `source ./scripts/start_neuron.sh` and follow prompts for validator. You can run this script with flags if you prefer not to enter prompts.

>[!NOTE]
> We recommend running with --logging.trace while we are in Beta. This is much more verbose, but it will help us
