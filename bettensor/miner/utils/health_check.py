import json
import os
import subprocess
import sys
import psycopg2
import time
import bittensor as bt

class HealthCheck:
    def __init__(self, db_params, axon_port):
        self.db_params = db_params
        self.axon_port = axon_port
        bt.logging.info(f"Initialized HealthCheck with db_params: {self.db_params}, axon_port: {self.axon_port}")

    def run_health_check(self):
        bt.logging.info("Starting health check")
        self.check_and_open_port(5000)  # Flask server
        self.check_and_open_port(9946)  # Subtensor
        self.check_and_open_port(self.axon_port)  # Axon port
        self.check_and_install_dependencies()
        self.check_and_setup_redis()
        self.check_and_setup_postgres()
        self.clear_pycache()
        self.check_and_update_python_deps()
        bt.logging.info("Health check completed")

    def run_command(self, command, error_message):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            bt.logging.info(f"Command executed successfully: {' '.join(command)}")
            return True
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"{error_message}: {e}")
            bt.logging.error(f"Command output: {e.output}")
            return False

    def check_and_open_port(self, port):
        if not self.run_command(['netstat', '-tuln'], f"Failed to check port {port}"):
            self.run_command(['sudo', 'ufw', 'allow', str(port)], f"Failed to open port {port}")

    def check_and_install_dependencies(self):
        bt.logging.info("Checking dependencies...")
        dependencies = ['build-essential', 'net-tools', 'clang', 'curl', 'git', 'make', 'libssl-dev', 'protobuf-compiler', 'llvm', 'libudev-dev', 'python-is-python3']
        missing_deps = []
        for dep in dependencies:
            if not self.run_command(['dpkg', '-s', dep], f"Checking {dep}"):
                missing_deps.append(dep)
        
        if missing_deps:
            bt.logging.info(f"Installing missing dependencies: {', '.join(missing_deps)}")
            self.run_command(['sudo', 'apt-get', 'update'], "Failed to update package list")
            self.run_command(['sudo', 'apt-get', 'install', '-y'] + missing_deps, "Failed to install dependencies")
        else:
            bt.logging.info("All dependencies are already installed.")

    def check_and_setup_redis(self):
        bt.logging.info("Checking Redis...")
        if not self.run_command(['redis-cli', 'ping'], "Redis is not running"):
            bt.logging.info("Setting up Redis...")
            if not self.run_command(['redis-server', '--version'], "Redis is not installed"):
                self.run_command(['sudo', 'apt-get', 'install', '-y', 'redis-server'], "Failed to install Redis")

            self.run_command(['sudo', 'systemctl', 'start', 'redis-server'], "Failed to start Redis")
            self.run_command(['sudo', 'systemctl', 'enable', 'redis-server'], "Failed to enable Redis")

            redis_conf = '/etc/redis/redis.conf'
            if os.path.exists(redis_conf):
                with open(redis_conf, 'r') as f:
                    config = f.read()
                if 'bind 0.0.0.0' not in config:
                    bt.logging.info("Modifying Redis configuration to allow connections from anywhere")
                    config = config.replace('bind 127.0.0.1', 'bind 0.0.0.0')
                    with open(redis_conf, 'w') as f:
                        f.write(config)
                    self.run_command(['sudo', 'systemctl', 'restart', 'redis-server'], "Failed to restart Redis")
        else:
            bt.logging.info("Redis is already set up and running.")

    def check_and_setup_postgres(self):
        bt.logging.info("Checking PostgreSQL...")
        bt.logging.info(f"Using database parameters: {self.db_params}")
        try:
            conn = psycopg2.connect(
                dbname=self.db_params['db_name'],
                user=self.db_params['db_user'],
                password=self.db_params['db_password'],
                host=self.db_params['db_host'],
                port=self.db_params['db_port']
            )
            conn.close()
            bt.logging.info("PostgreSQL is already set up and accessible.")
            return
        except psycopg2.OperationalError:
            bt.logging.info("Setting up PostgreSQL...")

        if not self.run_command(['psql', '--version'], "PostgreSQL is not installed"):
            self.run_command(['sudo', 'apt-get', 'install', '-y', 'postgresql', 'postgresql-contrib'], "Failed to install PostgreSQL")

        self.run_command(['sudo', 'systemctl', 'start', 'postgresql'], "Failed to start PostgreSQL")
        self.run_command(['sudo', 'systemctl', 'enable', 'postgresql'], "Failed to enable PostgreSQL")

        # Configure PostgreSQL
        commands = [
            "CREATE DATABASE {db_name} WITH OWNER = {db_user};",
            "CREATE USER {db_user} WITH SUPERUSER PASSWORD '{db_password}';",
            "ALTER USER {db_user} WITH SUPERUSER;",
            "GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};"
        ]
        for command in commands:
            formatted_command = command.format(**self.db_params)
            bt.logging.info(f"Executing PostgreSQL command: {formatted_command}")
            self.run_command(['sudo', '-u', 'postgres', 'psql', '-c', formatted_command], f"Failed to execute: {formatted_command}")

        # Modify PostgreSQL configuration
        pg_hba_conf = '/etc/postgresql/12/main/pg_hba.conf'
        postgresql_conf = '/etc/postgresql/12/main/postgresql.conf'
        
        self.run_command(['sudo', 'sed', '-i', 
                          's/local   all             postgres                                peer/local   all             postgres                                trust/', 
                          pg_hba_conf], "Failed to modify pg_hba.conf for postgres user")
        self.run_command(['sudo', 'sed', '-i', 
                          's/local   all             all                                     peer/local   all             all                                     trust/', 
                          pg_hba_conf], "Failed to modify pg_hba.conf for all users")
        self.run_command(['sudo', 'sed', '-i', 
                          "s/#listen_addresses = 'localhost'/listen_addresses = '*'/", 
                          postgresql_conf], "Failed to modify postgresql.conf")
        self.run_command(['sudo', 'bash', '-c', 
                          f'echo "host all all 0.0.0.0/0 md5" >> {pg_hba_conf}'], 
                         "Failed to add remote connections to pg_hba.conf")

        self.run_command(['sudo', 'systemctl', 'restart', 'postgresql'], "Failed to restart PostgreSQL")

    def check_and_update_python_deps(self):
        bt.logging.info("Checking Python dependencies...")
        required_packages = {
            'pip': '--upgrade pip',
            'psycopg2-binary': '--no-cache-dir psycopg2-binary',
            'torch': '--no-cache-dir torch==1.13.1',
            'bittensor': '--no-cache-dir bittensor==6.9.3'
        }
        
        for package, install_args in required_packages.items():
            if not self.run_command(['pip', 'show', package], f"{package} is not installed"):
                bt.logging.info(f"Installing {package}...")
                self.run_command(['pip', 'install'] + install_args.split(), f"Failed to install {package}")
            else:
                bt.logging.info(f"{package} is already installed.")
        
        bt.logging.info("Updating project requirements...")
        self.run_command(['pip', 'install', '-r', 'requirements.txt'], "Failed to install requirements")

    def clear_pycache(self):
        bt.logging.info("Clearing pycache...")
        try:
            # First, try to use find with -regextype posix-extended
            result = subprocess.run(
                ['find', '.', '-regextype', 'posix-extended', '-regex', '.*(__pycache__|\.pyc|\.pyo)$', '-exec', 'rm', '-rf', '{}', '+'],
                check=True, capture_output=True, text=True
            )
            bt.logging.info("Successfully cleared pycache using extended regex")
        except subprocess.CalledProcessError:
            bt.logging.warning("Failed to clear pycache using extended regex. Trying alternative method.")
            try:
                # If the first method fails, try a more basic approach
                subprocess.run(['find', '.', '-name', '__pycache__', '-exec', 'rm', '-rf', '{}', '+'], check=True)
                subprocess.run(['find', '.', '-name', '*.pyc', '-exec', 'rm', '-f', '{}', '+'], check=True)
                subprocess.run(['find', '.', '-name', '*.pyo', '-exec', 'rm', '-f', '{}', '+'], check=True)
                bt.logging.info("Successfully cleared pycache using alternative method")
            except subprocess.CalledProcessError as e:
                bt.logging.error(f"Failed to clear pycache: {e}")
                bt.logging.error(f"Command output: {e.output}")


def run_health_check(db_params, axon_port):
    bt.logging.info(f"Received db_params: {db_params}")
    bt.logging.info(f"Received axon_port: {axon_port}")
    health_check = HealthCheck(db_params, axon_port)
    health_check.run_health_check()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python health_check.py <axon_port> <db_params_json>")
        sys.exit(1)
    
    axon_port = int(sys.argv[1])
    db_params = json.loads(sys.argv[2])
    bt.logging.info(f"Parsed axon_port: {axon_port}")
    bt.logging.info(f"Parsed db_params: {db_params}")
    run_health_check(db_params, axon_port)