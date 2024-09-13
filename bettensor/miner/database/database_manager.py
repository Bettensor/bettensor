import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import bittensor as bt
import traceback
import os
import time

class DatabaseManager:
    def __init__(self, db_name, db_user, db_password, db_host='localhost', db_port=5432, max_connections=10):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.max_connections = max_connections
        
        bt.logging.debug("Initializing DatabaseManager")
        bt.logging.debug(f"Checking root user")
        self.is_root = self.check_root_user()
        bt.logging.debug(f"Ensuring database exists")
        self.ensure_database_exists()
        bt.logging.debug(f"Waiting for database")
        self.wait_for_database()
        bt.logging.debug(f"Creating connection pool")
        self.connection_pool = self.create_connection_pool()
        bt.logging.debug(f"Creating tables")
        self.create_tables()
        bt.logging.debug("DatabaseManager initialization complete")

    def check_root_user(self):
        return self.db_user == 'root'

    def ensure_database_exists(self):
        conn = None
        try:
            # Connect to the default 'postgres' database
            conn = psycopg2.connect(
                dbname='postgres',
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cur:
                # Check if the database exists
                cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.db_name,))
                exists = cur.fetchone()
                
                if not exists:
                    bt.logging.debug(f"Creating database {self.db_name}")
                    # Create the database
                    cur.execute(f"CREATE DATABASE {self.db_name}")
                    bt.logging.debug(f"Database {self.db_name} created successfully")
                else:
                    bt.logging.debug(f"Database {self.db_name} already exists")
        
        except psycopg2.Error as e:
            bt.logging.error(f"Error ensuring database exists: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def wait_for_database(self):
        max_retries = 5
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                with psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    user=self.db_user,
                    password=self.db_password,
                    database=self.db_name
                ) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        bt.logging.debug("Successfully connected to the database.")
                        return
            except psycopg2.OperationalError:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    bt.logging.error("Failed to connect to the database after multiple attempts.")
                    raise

    def create_connection_pool(self):
        return SimpleConnectionPool(
            1, self.max_connections,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def create_tables(self):
        tables = [
            ("predictions", """
            CREATE TABLE IF NOT EXISTS predictions (
                predictionID TEXT PRIMARY KEY, 
                teamGameID TEXT, 
                minerID TEXT, 
                predictionDate TEXT, 
                predictedOutcome TEXT,
                teamA TEXT,
                teamB TEXT,
                wager REAL,
                teamAodds REAL,
                teamBodds REAL,
                tieOdds REAL,
                outcome TEXT
            )
            """),
            ("games", """
            CREATE TABLE IF NOT EXISTS games (
                gameID TEXT PRIMARY KEY,
                teamA TEXT,
                teamAodds REAL,
                teamB TEXT,
                teamBodds REAL,
                sport TEXT,
                league TEXT,
                externalID TEXT UNIQUE,
                createDate TEXT,
                lastUpdateDate TEXT,
                eventStartDate TEXT,
                active INTEGER,
                outcome TEXT,
                tieOdds REAL,
                canTie BOOLEAN
            )
            """),
            ("miner_stats", """
            CREATE TABLE IF NOT EXISTS miner_stats (
                miner_hotkey TEXT PRIMARY KEY,
                miner_uid INTEGER,
                miner_rank INTEGER,
                miner_cash REAL,
                miner_current_incentive REAL,
                miner_last_prediction_date TEXT,
                miner_lifetime_earnings REAL,
                miner_lifetime_wager REAL,
                miner_lifetime_predictions INTEGER,
                miner_lifetime_wins INTEGER,
                miner_lifetime_losses INTEGER,
                miner_win_loss_ratio REAL,
                last_daily_reset TEXT
            )
            """),
            ("model_params", """
            CREATE TABLE IF NOT EXISTS model_params (
                id SERIAL PRIMARY KEY,
                model_on BOOLEAN,
                wager_distribution_steepness INTEGER,
                fuzzy_match_percentage INTEGER,
                minimum_wager_amount FLOAT,
                max_wager_amount FLOAT,
                top_n_games INTEGER
            )
            """),
            ("miner_active", """
            CREATE TABLE IF NOT EXISTS miner_active (
                miner_uid TEXT PRIMARY KEY,
                last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        ]
        
        for table_name, create_query in tables:
            try:
                self.execute_query(create_query)
                bt.logging.debug(f"Created table: {table_name}")
            except Exception as e:
                bt.logging.error(f"Error creating table {table_name}: {e}")

    def initialize_default_model_params(self, miner_uid):
        bt.logging.info(f"Initializing default model params for miner: {miner_uid}")
        self.ensure_model_params_table_exists()
        self.ensure_miner_params_exist(miner_uid)

    def ensure_model_params_table_exists(self):
        query = """
        CREATE TABLE IF NOT EXISTS model_params (
            id SERIAL PRIMARY KEY,
            model_on BOOLEAN,
            wager_distribution_steepness INTEGER,
            fuzzy_match_percentage INTEGER,
            minimum_wager_amount FLOAT,
            max_wager_amount FLOAT,
            top_n_games INTEGER,
            nfl_model_on BOOLEAN,
            nfl_minimum_wager_amount FLOAT,
            nfl_max_wager_amount FLOAT,
            nfl_top_n_games INTEGER,
            nfl_kelly_fraction_multiplier FLOAT,
            nfl_edge_threshold FLOAT,
            nfl_max_bet_percentage FLOAT
        )
        """
        self.execute_query(query)

    def ensure_miner_params_exist(self, miner_uid):
        default_params = {
            'model_on': False,
            'wager_distribution_steepness': 1,
            'fuzzy_match_percentage': 80,
            'minimum_wager_amount': 1.0,
            'max_wager_amount': 100.0,
            'top_n_games': 10,
            'nfl_model_on': False,
            'nfl_minimum_wager_amount': 1.0,
            'nfl_max_wager_amount': 100.0,
            'nfl_top_n_games': 5,
            'nfl_kelly_fraction_multiplier': 1,
            'nfl_edge_threshold': 0.02,
            'nfl_max_bet_percentage': 0.7
        }
        
        query = "SELECT * FROM model_params WHERE id = %s"
        result = self.execute_query(query, (miner_uid,))
        
        if not result:
            insert_query = """
            INSERT INTO model_params (
                id, model_on, wager_distribution_steepness, fuzzy_match_percentage,
                minimum_wager_amount, max_wager_amount, top_n_games,
                nfl_model_on, nfl_minimum_wager_amount, nfl_max_wager_amount,
                nfl_top_n_games, nfl_kelly_fraction_multiplier, nfl_edge_threshold,
                nfl_max_bet_percentage
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.execute_query(insert_query, (miner_uid, *default_params.values()))

    def get_model_params(self, miner_uid):
        query = "SELECT * FROM model_params WHERE id = %s"
        result = self.execute_query(query, (miner_uid,))
        return result[0] if result else None

    def update_model_params(self, miner_uid, params):
        query = """
        UPDATE model_params SET
            model_on = %s,
            wager_distribution_steepness = %s,
            fuzzy_match_percentage = %s,
            minimum_wager_amount = %s,
            max_wager_amount = %s,
            top_n_games = %s,
            nfl_model_on = %s,
            nfl_minimum_wager_amount = %s,
            nfl_max_wager_amount = %s,
            nfl_top_n_games = %s,
            nfl_kelly_fraction_multiplier = %s,
            nfl_edge_threshold = %s,
            nfl_max_bet_percentage = %s
        WHERE id = %s
        """
        self.execute_query(query, (*params.values(), miner_uid))

    def execute_query(self, query, params=None):
        # print(f"DatabaseManager: Executing query: {query}")
        # print(f"DatabaseManager: Query parameters: {params}")
        try:
            with self.connection_pool.getconn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    if query.strip().upper().startswith("SELECT"):
                        result = cur.fetchall()
                    else:
                        result = cur.rowcount
                        conn.commit()
                    # print(f"DatabaseManager: Query result: {result}")
                    return result
        except Exception as e:
            # print(f"DatabaseManager: Error executing query: {str(e)}")
            # print(f"DatabaseManager: Traceback: {traceback.format_exc()}")
            raise
        finally:
            self.connection_pool.putconn(conn)

    def execute_batch(self, query, params_list):
        conn, cur = None, None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            bt.logging.debug(f"Executing batch query: {query}")
            bt.logging.debug(f"Number of parameter sets: {len(params_list)}")
            
            cur.executemany(query, params_list)
            conn.commit()
            bt.logging.debug("Batch query executed successfully")
        except Exception as e:
            if conn:
                conn.rollback()
            bt.logging.error(f"Error in execute_batch: {str(e)}")
            bt.logging.error(f"Query: {query}")
            bt.logging.error(f"Params: {params_list}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    def close(self):
        self.connection_pool.closeall()

    def ensure_miner_active_table_exists(self):
        query = """
        CREATE TABLE IF NOT EXISTS miner_active (
            miner_uid TEXT PRIMARY KEY,
            last_active_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.execute_query(query)
    def update_miner_activity(self, miner_uid):
        query = """
        INSERT INTO miner_active (miner_uid, last_active_timestamp)
        VALUES (%s, CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
        ON CONFLICT (miner_uid) DO UPDATE
        SET last_active_timestamp = CURRENT_TIMESTAMP AT TIME ZONE 'UTC'
        """
        self.execute_query(query, (miner_uid,))