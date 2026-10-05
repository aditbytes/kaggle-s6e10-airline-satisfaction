PY := .venv/bin/python
COMP := playground-series-s6e10
EXP ?= exp001_lgbm_raw
FEATURES ?= raw

.PHONY: setup data eda smoke train submit

setup:  ## create .venv and install dependencies
	uv venv --relocatable --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt

data:  ## download and unzip the competition data (accept the rules on kaggle.com first)
	kaggle competitions download $(COMP) -p data
	unzip -o data/$(COMP).zip -d data && rm data/$(COMP).zip

eda:  ## regenerate reports/eda.md
	$(PY) src/eda.py

smoke:  ## 30-second check that training works
	$(PY) src/train.py --exp smoke --quick

train:  ## full CV run: make train EXP=exp002_lgbm_fe FEATURES=fe
	$(PY) -u src/train.py --exp $(EXP) --features $(FEATURES)

submit:  ## build submissions/$(EXP).csv
	$(PY) src/submit.py $(EXP)
