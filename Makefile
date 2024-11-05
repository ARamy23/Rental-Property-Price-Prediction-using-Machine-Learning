# Variables
DB_NAME = rental_db
DB_USER = postgres
SQL_SETUP_FILE = setup.sql
CONDA_ENV_NAME = rental_env
STREAMLIT_APP = app.py  # Replace with the actual Streamlit app file if different

# Default target: Sets up the whole environment
all: install_conda install_dependencies setup_db start_streamlit

# Step 1: Install Miniconda if not already installed
install_conda:
	curl -o Miniconda3-latest-MacOSX-arm64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
	bash Miniconda3-latest-MacOSX-arm64.sh -b -p $$HOME/miniconda
	export PATH="$$HOME/miniconda/bin:$$PATH"
	source $$HOME/miniconda/bin/activate
	conda init

# Step 2: Create and activate Conda environment, install dependencies
install_dependencies:
	conda create -n $(CONDA_ENV_NAME) python=3.9 -y
	conda activate $(CONDA_ENV_NAME) && conda install -c conda-forge psycopg2-binary pandas streamlit -y

# Step 3: Start PostgreSQL service if using Homebrew (for macOS)
start_postgres:
	brew services start postgresql

# Step 4: Create the database and run the SQL setup script
setup_db: start_postgres
	psql -U $(DB_USER) -c "DROP DATABASE IF EXISTS $(DB_NAME);"
	psql -U $(DB_USER) -c "CREATE DATABASE $(DB_NAME);"
	psql -U $(DB_USER) -d $(DB_NAME) -f $(SQL_SETUP_FILE)

# Step 5: Start Streamlit application (model interface)
start_streamlit:
	conda activate $(CONDA_ENV_NAME) && streamlit run $(STREAMLIT_APP)

# Helper target: Quick start of Conda environment
activate_env:
	@echo "Activating Conda environment '$(CONDA_ENV_NAME)'..."
	conda activate $(CONDA_ENV_NAME)

# Clean up: Remove downloaded Miniconda installer
clean:
	rm -f Miniconda3-latest-MacOSX-arm64.sh

.PHONY: all install_conda install_dependencies start_postgres setup_db start_streamlit activate_env clean
