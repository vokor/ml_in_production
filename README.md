# ml_in_production

This is sample project of ml production ready code.

I've used dataset from here: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/code

Project contains several directories:
* ```config```: all .yaml files, specifically ```train_config.yaml``` with training parameters
* ```dataset```: all datasets
* ```ml_core```: responsible for training and validation
* ```notebooks```: all research

---

### Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

* Train
```python ml_core/main.py train --train-config ../configs/train_config.yaml --logging-config ../configs/logger.yaml```
* Eval
* ```python ml_core/main.py eval -i models/model.pkl -d ../dataset/test_data_mini.csv -o result.csv```

PS: you may have problems due to running from console (I'm trying to fix this). In PyCharm everything is OK...

I think I've dove: 0, 1, 2, 3, 4, 5, 6, 9 (1-2 points because only one config), 10, 11 (used transformer but not test it, maybe 1 point), 12