# 一条命令复现：make all（下载语料 → 切段 → 校验金标 → 建索引并评测 → 消融 → 出表）
PYTHON ?= python

.PHONY: all data gold eval ablate report test clean

all: data gold eval ablate report

data:
	$(PYTHON) data/fetch_edgar.py
	$(PYTHON) data/fetch_wiki.py
	$(PYTHON) data/build_passages.py

gold:
	$(PYTHON) goldset/build_gold.py --check

# 第一次运行会编码全部段落（向量缓存在 .work/），之后复用缓存
eval:
	$(PYTHON) -m rre.run_eval

ablate:
	$(PYTHON) -m rre.ablate_prefix

report:
	$(PYTHON) -m rre.report

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf .work
