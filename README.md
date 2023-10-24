# neural_networks



## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (Version you've developed on, e.g., Python 3.11+)
- [Poetry](https://python-poetry.org/docs/): A dependency management and packaging tool.



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



# Run with docker

by default it will run all configurations in configurations.toml

We recomand to mound the data, results and logs folder externally for easier access, if you have a configurations.toml that you want to try, you can mount also the config folder and put the file there


```
docker run -it \
    -v ~/Documents/neural_networks/data:/data \
    -v ~/Documents/neural_networks/results:/app/results \
    -v ~/Documents/neural_networks/logs:/app/logs \
    -p 6006:6006 \
    pmallozzi/neural_networks:latest
```

you can access tensorboard at localhost:6000

passing one specific configuration

```
docker run -it \
    -v ~/Documents/neural_networks/data:/data \
    -v ~/Documents/neural_networks/results:/app/results \
    -v ~/Documents/neural_networks/logs:/app/logs \
    -v ~/Documents/neural_networks/config:/app/config \
    -p 6006:6006 \
    pmallozzi/neural_networks:latest CONFIGS='config_A'
```

passing list of configurations
```
docker run -it \
    -v ~/Documents/neural_networks/data:/data \
    -v ~/Documents/neural_networks/results:/app/results \
    -v ~/Documents/neural_networks/logs:/app/logs \
    -v ~/Documents/neural_networks/config:/app/config \
    -p 6006:6006 \
    pmallozzi/neural_networks:latest CONFIGS='config_A config_P'
```

running in the background

```
docker run -d \
    -- my_running_container
    -v ~/Documents/neural_networks/data:/data \
    -v ~/Documents/neural_networks/results:/app/results \
    -v ~/Documents/neural_networks/logs:/app/logs \
    -v ~/Documents/neural_networks/config:/app/config \
    -p 6006:6006 \
    pmallozzi/neural_networks:latest
```

you can see the live logs of the container:

docker logs -f my_running_container
