# mlops_m6_project

Run the `src/mlops-m6-repo/train.py` script to train a model and save it to the `models/` directory. Use the command below:

```bash
uv run src/mlops_m6_project/train.py --epochs 10 --batch-size 64 --lr 0.001
```

To evaluate the trained model, run the `src/mlops-m6-repo/evaluate.py` script with the following command:

```bash
uv run src/mlops_m6_project/evaluate.py --model-checkpoint models/model.pth
```

Finally, you can visualize how the model embeds the data using the `src/mlops-m6-repo/visualize.py` script:

```bash
uv run src/mlops_m6_project/visualize.py --model-checkpoint models/model.pth
```

To run the dockerfiles for training and evaluation, use the following commands:
```bash
cd mlops-m6-repo

docker build --platform linux/amd64 -f ./dockerfiles/train.dockerfile . -t train:latest

docker run --name experiment2 -v "$(pwd)/models:/models" -v "$(pwd)/reports:/reports" train:latest
```

```bash
cd mlops-m6-repo

docker build --platform linux/amd64 -f ./dockerfiles/evaluate.dockerfile . -t evaluate:latest

docker run --name evaluate --rm \
-v "$(pwd)/models/model.pth:/models/model.pth" \
-v "$(pwd)/data:/data" \
evaluate:latest
```

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
