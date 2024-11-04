"""
database_init.py

This file contains the code for initializing the database for the validator.
"""

def initialize_database():
    return [
        # Version tracking table
        """
        CREATE TABLE IF NOT EXISTS db_version (
            version INTEGER PRIMARY KEY
        )
        """,
        
        # Create miner_stats table
        """
        CREATE TABLE IF NOT EXISTS miner_stats (
            miner_hotkey TEXT PRIMARY KEY,
            miner_coldkey TEXT,
            miner_uid INTEGER UNIQUE,
            miner_rank INTEGER,
            miner_status TEXT,
            miner_cash REAL,
            miner_current_incentive REAL,
            miner_current_tier INTEGER,
            miner_current_scoring_window INTEGER,
            miner_current_composite_score REAL,
            miner_current_entropy_score REAL,
            miner_current_sharpe_ratio REAL,
            miner_current_sortino_ratio REAL,
            miner_current_roi REAL,
            miner_current_clv_avg REAL,
            miner_last_prediction_date TEXT,
            miner_lifetime_earnings REAL,
            miner_lifetime_wager_amount REAL,
            miner_lifetime_roi REAL,
            miner_lifetime_predictions INTEGER,
            miner_lifetime_wins INTEGER,
            miner_lifetime_losses INTEGER,
            miner_win_loss_ratio REAL
        )
        """,

        # Create predictions table with all columns
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            game_id INTEGER,
            miner_uid INTEGER,
            prediction_date TEXT,
            predicted_outcome INTEGER,
            predicted_odds REAL,
            team_a TEXT,
            team_b TEXT,
            wager REAL,
            team_a_odds REAL,
            team_b_odds REAL,
            tie_odds REAL,
            model_name TEXT,
            confidence_score REAL,
            outcome INTEGER,
            payout REAL,
            sent_to_site INTEGER DEFAULT 0,
            validators_sent_to INTEGER DEFAULT 0,
            validators_confirmed INTEGER DEFAULT 0
        )
        """,

        # Create game_data table
        """
        CREATE TABLE IF NOT EXISTS game_data (
            game_id TEXT PRIMARY KEY UNIQUE,
            external_id INTEGER UNIQUE,
            team_a TEXT,
            team_b TEXT,
            team_a_odds REAL,
            team_b_odds REAL,
            tie_odds REAL,
            can_tie BOOLEAN,
            event_start_date TEXT,
            create_date TEXT,
            last_update_date TEXT,
            sport TEXT,
            league TEXT,
            outcome INTEGER DEFAULT 3,
            active INTEGER
        )
        """,

        # Create keys table
        """
        CREATE TABLE IF NOT EXISTS keys (
            hotkey TEXT PRIMARY KEY,
            coldkey TEXT
        )
        """,

        # Create scores table
        """
        CREATE TABLE IF NOT EXISTS scores (
            miner_uid INTEGER,
            day_id INTEGER,
            score_type TEXT,
            clv_score REAL,
            roi_score REAL,
            sortino_score REAL,
            entropy_score REAL,
            composite_score REAL,
            PRIMARY KEY (miner_uid, day_id, score_type),
            FOREIGN KEY (miner_uid) REFERENCES miner_stats(miner_uid)
        )
        """,

        # Create score_state table
        """
        CREATE TABLE IF NOT EXISTS score_state (
            state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_day INTEGER,
            current_date TEXT,
            reference_date TEXT,
            invalid_uids TEXT,
            valid_uids TEXT,
            tiers TEXT,
            amount_wagered TEXT,
            last_update_date TEXT
        )
        """,


        # Create cleanup triggers
        """
        CREATE TRIGGER IF NOT EXISTS delete_old_predictions
        AFTER INSERT ON predictions
        BEGIN
            DELETE FROM predictions
            WHERE prediction_date < date('now', '-50 days');
        END;
        """,

        """
        CREATE TRIGGER IF NOT EXISTS delete_old_game_data
        AFTER INSERT ON game_data
        BEGIN
            DELETE FROM game_data
            WHERE event_start_date < date('now', '-50 days');
        END;
        """,

        """
        CREATE TRIGGER IF NOT EXISTS delete_old_score_state
        AFTER INSERT ON score_state
        BEGIN
            DELETE FROM score_state
            WHERE last_update_date < date('now', '-7 days');
        END;
        """,

        # Initialize version if not exists
        """
        INSERT OR IGNORE INTO db_version (version) VALUES (1)
        """
    ]
