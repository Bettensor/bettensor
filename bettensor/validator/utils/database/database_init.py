"""
database_init.py

This file contains the code for initializing the database for the validator.
"""

def initialize_database():
    statements = []
    
    # 1. Disable foreign key constraints (optional but recommended)
    statements.append("PRAGMA foreign_keys = OFF;")
    
    # 2. Create main miner_stats table first
    statements.append("""
        CREATE TABLE IF NOT EXISTS miner_stats (
            miner_uid INTEGER PRIMARY KEY,
            miner_hotkey TEXT UNIQUE,
            miner_coldkey TEXT,
            miner_rank INTEGER DEFAULT 0,
            miner_status TEXT DEFAULT 'active',
            miner_cash REAL DEFAULT 0.0,
            miner_current_incentive REAL DEFAULT 0.0,
            miner_current_tier INTEGER DEFAULT 1,
            miner_current_scoring_window INTEGER DEFAULT 0,
            miner_current_composite_score REAL DEFAULT 0.0,
            miner_current_entropy_score REAL DEFAULT 0.0,
            miner_current_sharpe_ratio REAL DEFAULT 0.0,
            miner_current_sortino_ratio REAL DEFAULT 0.0,
            miner_current_roi REAL DEFAULT 0.0,
            miner_current_clv_avg REAL DEFAULT 0.0,
            miner_last_prediction_date TEXT,
            miner_lifetime_earnings REAL DEFAULT 0.0,
            miner_lifetime_wager_amount REAL DEFAULT 0.0,
            miner_lifetime_roi REAL DEFAULT 0.0,
            miner_lifetime_predictions INTEGER DEFAULT 0,
            miner_lifetime_wins INTEGER DEFAULT 0,
            miner_lifetime_losses INTEGER DEFAULT 0,
            miner_win_loss_ratio REAL DEFAULT 0.0
        )
    """)
    
    # 3. Create backup table
    statements.append("""
        CREATE TABLE IF NOT EXISTS miner_stats_backup (
            miner_uid INTEGER,
            miner_hotkey TEXT,
            miner_coldkey TEXT,
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
    """)
    
    # 4. Backup existing data with proper casting
    statements.append("""
        INSERT OR IGNORE INTO miner_stats_backup 
        SELECT 
            CAST(miner_uid AS INTEGER),
            miner_hotkey,
            miner_coldkey,
            CAST(COALESCE(miner_rank, 0) AS INTEGER),
            COALESCE(miner_status, 'active'),
            CAST(COALESCE(miner_cash, 0.0) AS REAL),
            CAST(COALESCE(miner_current_incentive, 0.0) AS REAL),
            CAST(COALESCE(miner_current_tier, 1) AS INTEGER),
            CAST(COALESCE(miner_current_scoring_window, 0) AS INTEGER),
            CAST(COALESCE(miner_current_composite_score, 0.0) AS REAL),
            CAST(COALESCE(miner_current_entropy_score, 0.0) AS REAL),
            CAST(COALESCE(miner_current_sharpe_ratio, 0.0) AS REAL),
            CAST(COALESCE(miner_current_sortino_ratio, 0.0) AS REAL),
            CAST(COALESCE(miner_current_roi, 0.0) AS REAL),
            CAST(COALESCE(miner_current_clv_avg, 0.0) AS REAL),
            miner_last_prediction_date,
            CAST(COALESCE(miner_lifetime_earnings, 0.0) AS REAL),
            CAST(COALESCE(miner_lifetime_wager_amount, 0.0) AS REAL),
            CAST(COALESCE(miner_lifetime_roi, 0.0) AS REAL),
            CAST(COALESCE(miner_lifetime_predictions, 0) AS INTEGER),
            CAST(COALESCE(miner_lifetime_wins, 0) AS INTEGER),
            CAST(COALESCE(miner_lifetime_losses, 0) AS INTEGER),
            CAST(COALESCE(miner_win_loss_ratio, 0.0) AS REAL)
        FROM miner_stats WHERE EXISTS (SELECT 1 FROM miner_stats)
    """)
    
    # 5. Restore from backup with proper casting
    statements.append("""
        INSERT OR REPLACE INTO miner_stats 
        SELECT 
            CAST(miner_uid AS INTEGER) as miner_uid,
            miner_hotkey,
            miner_coldkey,
            CAST(COALESCE(miner_rank, 0) AS INTEGER) as miner_rank,
            COALESCE(miner_status, 'active') as miner_status,
            CAST(COALESCE(miner_cash, 0.0) AS REAL) as miner_cash,
            CAST(COALESCE(miner_current_incentive, 0.0) AS REAL) as miner_current_incentive,
            CAST(COALESCE(miner_current_tier, 1) AS INTEGER) as miner_current_tier,
            CAST(COALESCE(miner_current_scoring_window, 0) AS INTEGER) as miner_current_scoring_window,
            CAST(COALESCE(miner_current_composite_score, 0.0) AS REAL) as miner_current_composite_score,
            CAST(COALESCE(miner_current_entropy_score, 0.0) AS REAL) as miner_current_entropy_score,
            CAST(COALESCE(miner_current_sharpe_ratio, 0.0) AS REAL) as miner_current_sharpe_ratio,
            CAST(COALESCE(miner_current_sortino_ratio, 0.0) AS REAL) as miner_current_sortino_ratio,
            CAST(COALESCE(miner_current_roi, 0.0) AS REAL) as miner_current_roi,
            CAST(COALESCE(miner_current_clv_avg, 0.0) AS REAL) as miner_current_clv_avg,
            miner_last_prediction_date,
            CAST(COALESCE(miner_lifetime_earnings, 0.0) AS REAL) as miner_lifetime_earnings,
            CAST(COALESCE(miner_lifetime_wager_amount, 0.0) AS REAL) as miner_lifetime_wager_amount,
            CAST(COALESCE(miner_lifetime_roi, 0.0) AS REAL) as miner_lifetime_roi,
            CAST(COALESCE(miner_lifetime_predictions, 0) AS INTEGER) as miner_lifetime_predictions,
            CAST(COALESCE(miner_lifetime_wins, 0) AS INTEGER) as miner_lifetime_wins,
            CAST(COALESCE(miner_lifetime_losses, 0) AS INTEGER) as miner_lifetime_losses,
            CAST(COALESCE(miner_win_loss_ratio, 0.0) AS REAL) as miner_win_loss_ratio
        FROM miner_stats_backup 
        WHERE EXISTS (SELECT 1 FROM miner_stats_backup)
    """)
    
    # 6. Create dependent tables
    statements.extend([
        # Create predictions table
        """CREATE TABLE IF NOT EXISTS predictions (
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
        )""",
        
        # Create game_data table
        """CREATE TABLE IF NOT EXISTS game_data (
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
        )""",
        
        # Create keys table
        """CREATE TABLE IF NOT EXISTS keys (
            hotkey TEXT PRIMARY KEY,
            coldkey TEXT
        )""",
        
        # Create scores table
        """CREATE TABLE IF NOT EXISTS scores (
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
        )""",
        
        # Create score_state table
        """CREATE TABLE IF NOT EXISTS score_state (
            state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_day INTEGER,
            current_date TEXT,
            reference_date TEXT,
            invalid_uids TEXT,
            valid_uids TEXT,
            tiers TEXT,
            amount_wagered TEXT,
            last_update_date TEXT
        )""",
    ])
    
    # 7. Create triggers
    statements.extend([
        """CREATE TRIGGER IF NOT EXISTS delete_old_predictions
        AFTER INSERT ON predictions
        BEGIN
            DELETE FROM predictions
            WHERE prediction_date < date('now', '-50 days');
        END""",
        
        """CREATE TRIGGER IF NOT EXISTS delete_old_game_data
        AFTER INSERT ON game_data
        BEGIN
            DELETE FROM game_data
            WHERE event_start_date < date('now', '-50 days');
        END""",
        
        """CREATE TRIGGER IF NOT EXISTS delete_old_score_state
        AFTER INSERT ON score_state
        BEGIN
            DELETE FROM score_state
            WHERE last_update_date < date('now', '-7 days');
        END""",
    ])

    # 8. Re-enable foreign key constraints
    statements.append("PRAGMA foreign_keys = ON;")
    
    # 9. Initialize version
    statements.append(
        """INSERT OR IGNORE INTO db_version (version) VALUES (1)"""
    )
    
    return statements
