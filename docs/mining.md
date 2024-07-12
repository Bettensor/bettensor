# Bettensor Miner Guide

This guide provides detailed information for setting up and running a Bettensor miner. Whether you're new to the network or an experienced user, this document will help you get started and make the most of your mining experience.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Miner Setup](#miner-setup)
3. [Choosing Your Interface](#choosing-your-interface)
4. [Running Your Miner](#running-your-miner)
5. [Submitting Predictions](#submitting-predictions)
6. [Managing Multiple Miners](#managing-multiple-miners)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Frequently Asked Questions](#frequently-asked-questions)

## Getting Started

Before you begin, ensure you have the following prerequisites:

- Python 3.8 or higher installed
- Git installed
- Basic knowledge of command-line operations and/or your favorite llm chatbot to guide you along


>[!WARNING]
> This guide is being updated to use the new interfaces. Please ignore any references to the Central Server Interface, for the time being. This functionality is still being developed and will be available in the near future.

Clone the Bettensor repository and install the required dependencies:

```bash
git clone https://github.com/bettensor/bettensor.git
cd bettensor
pip install -r requirements.txt
```

## Miner Setup

1. Navigate to the Bettensor directory:
   ```bash
   cd path/to/bettensor
   ```

2. Run the start neuron script:
   ```bash
   source ./scripts/start_neuron.sh
   ```

3. Follow the prompts to set up your miner. You can also use command-line flags to skip the prompts:
   ```bash
   source ./scripts/start_neuron.sh --netuid 1 --subtensor.network test --wallet.name mywallet --wallet.hotkey myhotkey
   ```

>[!NOTE]
> We recommend running with --logging.trace while we are in Beta. This is much more verbose, but it will help us to pinpoint and fix any issues more quickly.




<details>
<summary>
Coming Soon... New Interfaces
</summary>
## Choosing Your Interface

When setting up your miner, you'll be prompted to choose between two interface options:

### Local Interface

The Local Interface runs on your local machine and is not accessible from the internet.

**Choose Local Interface if:**
- You prioritize privacy and want to keep your miner isolated from external connections.
- You're comfortable with a more basic user interface and don't need advanced features.
- You prefer to manage your miner(s) directly on your local machine.

### Central Server

The Central Server option connects your miner to our web dashboard, allowing for remote management and access to advanced features.

**Choose Central Server if:**
- You want a more user-friendly and feature-rich experience.
- You're comfortable with your miner connecting to our secure central server.
- You want to access and manage your miner(s) from any device through our web dashboard.
- You're interested in more comprehensive data analysis and visualization tools.

>[!IMPORTANT]
> The Central Server option provides a more streamlined experience and shows more comprehensive data than the local interface. However, it does require your miner to accept connections from our server.


</details>


## Running Your Miner

After setup, your miner will start automatically. You can check the logs to ensure it's running correctly:

```bash
pm2 logs miner
```

Wait for some game data to be received before proceeding to submit predictions.

## Submitting Predictions

### Local Interface

If you chose the Local Interface, use the CLI to submit predictions:

```bash
python bettensor/miner/cli.py
```


<details>
<summary>
Coming Soon... New Interfaces
</summary>
### Central Server

If you chose the Central Server option, log in to our [web dashboard](https://bettensor.com/dashboard) to connect your miner and submit predictions.
</details>

## Managing Multiple Miners

### Local Interface



When running multiple miners locally, use the `--uid` flag to specify which miner you want to interact with:

```bash
python bettensor/miner/cli.py --uid <UID>
```

You can find your miner UIDs in the `./data/miner_env.txt` file.

### Central Server

When using the Central Server option, you can manage multiple miners directly from the web dashboard without needing to specify UIDs manually.

## Security Considerations

- The Local Interface provides an additional layer of security by running only on your local machine.
- The Central Server option uses secure connections (SSL/TLS) and authentication tokens to protect your data.
- Always use the latest version of the software and keep your system updated.
- Be cautious when using public or unsecured networks, especially when accessing the web dashboard.

## Troubleshooting

If you encounter issues:

1. Check the logs for error messages:
   ```bash
   pm2 logs <miner process name>
   ```

2. Ensure your system meets all prerequisites and dependencies are correctly installed.

3. Verify your network connection and firewall settings, especially if using the Central Server option.

4. If problems persist, please open an issue on our GitHub repository or contact our support team.

## Frequently Asked Questions

**Q: Can I switch between Local Interface and Central Server after initial setup?**

A: Yes, you can change your interface type by running the start_neuron.sh script again and selecting a different option.

**Q: Is my data safe when using the Central Server option?**

A: We employ industry-standard security measures to protect your data. However, as with any internet-connected service, there's always a small risk. Choose the option you're most comfortable with.

**Q: How often should I submit predictions?**

A: The frequency of predictions can vary based on network activity and your strategy. Monitor the network and adjust accordingly.

**Q: Can I run miners on multiple machines?**

A: Yes, you can run miners on different machines. Each miner will need its own wallet and hotkey.

For more questions or support, please visit our [community forum](https://community.bettensor.com) or [Discord channel](https://discord.gg/bettensor).
