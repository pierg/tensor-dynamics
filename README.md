# neural_networks



## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (Version you've developed on, e.g., Python 3.11+)
- [Poetry](https://python-poetry.org/docs/): A dependency management and packaging tool.



## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/pierg/neural_networks.git
   cd neural_networks
   ```

2. **Install Dependencies**

   With Poetry installed, run the following command:

   ```bash
   poetry config virtualenvs.in-project true
   poetry install
   ```

   This will read the `pyproject.toml` and install all necessary dependencies.



# Run with docker

by default it will run all configurations in configurations.toml

```
docker run -it -v ~/Documents/data:/data -v ~/Documents/results:/app/results -p 6006:6006 pmallozzi/neural_networks:latest
```

passing one configuration
```
docker run -it -v ~/Documents/data:/data -v ~/Documents/results:/app/results -p 6006:6006 pmallozzi/neural_networks:latest CONFIGS='config_A'
```

passing list of configurations
```
docker run -it -v ~/Documents/data:/data -v ~/Documents/results:/app/results -p 6006:6006 pmallozzi/neural_networks:latest CONFIGS='config_A config_P'
```