# neural_networks



## Prerequisites

Before you begin, ensure you have met the following requirements:

- Tensorflow
- Python (Version you've developed on, e.g., Python 3.11+)


To install dependencies:

You can uyse poetry
- [Poetry](https://python-poetry.org/docs/): A dependency management and packaging tool.

Or just requirements.txt




## Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone --depth 1 https://github.com/pierg/neural_networks.git
   cd neural_networks
   ```

2. **Install Dependencies**

   With Poetry installed, run the following command:

   ```bash
   poetry config virtualenvs.in-project true
   poetry install
   ```


   This will read the `pyproject.toml` and install all necessary dependencies.

   then do "poetry shell"
   and then pip install tensordflow

   
3. ** Run **
   poetry python src/main.py


# Run with docker

by default it will run all configurations in configurations.toml


```
docker run -it \
    -v <LOCAL_RESULTS_PATH>:/app/results \
    pmallozzi/neural_networks:latest
```


you can run a specific configuration

```
docker run -it \
    -v <LOCAL_RESULTS_PATH>:/app/results \
    pmallozzi/neural_networks:latest CONFIGS='E'
```



or pass a list of configurations

```
docker run -it \
    -v ~/Documents/neural_networks/data:/data \
    -v ~/Documents/neural_networks/results:/app/results \
    -v ~/Documents/neural_networks/logs:/app/logs \
    -v ~/Documents/neural_networks/config:/app/config \
    -p 6006:6006 \
    pmallozzi/neural_networks:latest CONFIGS='E A'
```

the id of the confugrationa are the ids in the configurations.toml
