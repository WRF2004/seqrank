PYTHON ?= python
DATA ?= data/prepared
RUN ?= runs/demo

.PHONY: test lint demo serve
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src tests

demo:
	$(PYTHON) -m seqrank.cli demo-data
	$(PYTHON) -m seqrank.cli prepare --source data/events.csv --out $(DATA) --train-end 7000 --valid-end 8500 --max-len 20 --shard-size 1000
	$(PYTHON) -m seqrank.cli train --data $(DATA) --out $(RUN) --stage retrieval
	$(PYTHON) -m seqrank.cli train --data $(DATA) --out $(RUN) --stage ranker
	$(PYTHON) -m seqrank.cli evaluate --data $(DATA) --retrieval $(RUN)/retrieval-best.pt --ranker $(RUN)/ranker-best.pt --out $(RUN)/metrics.json

serve:
	$(PYTHON) -m seqrank.cli serve --data $(DATA) --retrieval $(RUN)/retrieval-best.pt --ranker $(RUN)/ranker-best.pt
