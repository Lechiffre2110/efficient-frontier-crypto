# Efficient Frontier Calculation API

This project is a simple Flask API that calculates the Efficient Frontier for a portfolio of stocks & crypto currencies, using Yahoo Finance to gather historical data.

## Requirements

You'll need to have the following installed on your system:

- Python 3.7 or newer
- pip, the Python package installer

## Installation

1. First, clone the repository to your local machine using git or download the zip file.
2. Next, navigate to the project directory and install the required packages using pip.

```bash
cd efficient-frontier-crypto
pip install -r requirements.txt
```

## Usage

### Running the API Locally

To run the API locally, navigate to the directory containing the `data.py` file, then run:

```bash
python data.py
```

This will start the Flask development server on your local machine. You can then send requests to the API using the react frontend or a tool like Postman.

