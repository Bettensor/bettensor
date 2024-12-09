import asyncio
import os
from bettensor.validator.utils.state_sync import StateSync

async def main():
    # Initialize state sync
    state_sync = StateSync()
    
    # Set target path in project root
    target_path = os.path.join(os.getcwd(), "validator.db")
    
    # Download the database
    success = await state_sync.download_single_file("validator.db", target_path)
    
    if success:
        print(f"Successfully downloaded validator.db to {target_path}")
    else:
        print("Failed to download validator.db")

if __name__ == "__main__":
    asyncio.run(main()) 