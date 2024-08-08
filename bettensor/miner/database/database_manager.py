import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import bittensor as bt
import traceback

class DatabaseManager:
    def __init__(self, db_name, db_user, db_password, db_host='localhost', db_port=5432, max_connections=10):
        self.connection_pool = SimpleConnectionPool(
            1, max_connections,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

    def execute_query(self, query, params=None):
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                bt.logging.debug(f"Executing query: {query}")
                bt.logging.debug(f"Query params: {params}")
                
                cur.execute(query, params)
                
                if query.strip().upper().startswith("SELECT"):
                    result = cur.fetchall()
                    bt.logging.debug(f"Query result: {result}")
                    return result
                else:
                    conn.commit()
                    bt.logging.debug("Query executed successfully")
                    return None
        except Exception as e:
            conn.rollback()
            bt.logging.error(f"Error in execute_query: {str(e)}")
            bt.logging.error(f"Query: {query}")
            bt.logging.error(f"Params: {params}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            self.connection_pool.putconn(conn)

    def execute_batch(self, query, params_list):
        conn = self.connection_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                bt.logging.debug(f"Executing batch query: {query}")
                bt.logging.debug(f"Number of parameter sets: {len(params_list)}")
                
                cur.executemany(query, params_list)
                conn.commit()
                bt.logging.debug("Batch query executed successfully")
        except Exception as e:
            conn.rollback()
            bt.logging.error(f"Error in execute_batch: {str(e)}")
            bt.logging.error(f"Query: {query}")
            bt.logging.error(f"Params: {params_list}")
            bt.logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            self.connection_pool.putconn(conn)

    def close(self):
        self.connection_pool.closeall()