# Experiment pipeline. Usage: make pipeline EXP=base, make smoke.
EXP ?= base
REF ?= ae
GO  ?= go
UV  ?= uv
GOFMT ?= gofmt
DEVICE ?= auto
OUT := out/$(EXP)
AE  := cd autoencoder && $(UV) run --locked python -m graph_ae

.PHONY: test test-go test-python check data markov recover ae ae-resume ae-bench evaluate pipeline smoke figures analysis verify-results

test: test-go test-python

test-go:
	$(GO) test ./...

test-python:
	cd autoencoder && $(UV) run --locked pytest -q

check:
	@test -z "$$($(GOFMT) -l cmd internal)" || { $(GOFMT) -l cmd internal; exit 1; }
	$(GO) vet ./...
	cd autoencoder && $(UV) run --locked ruff check graph_ae tests ../scripts
	cd autoencoder && $(UV) run --locked ruff format --check graph_ae tests ../scripts

data:
	$(GO) run ./cmd/dataset -config experiments/$(EXP).json -out $(OUT)

markov:
	$(GO) run ./cmd/markov-train -experiment $(OUT) -max-order 5

recover:
	$(GO) run ./cmd/recover -experiment $(OUT) -method random
	$(GO) run ./cmd/recover -experiment $(OUT) -method markov -max-order 5

ae:
	$(AE) train --experiment ../$(OUT) --device $(DEVICE)
	$(AE) predict --experiment ../$(OUT) --device $(DEVICE)

ae-resume:
	$(AE) train --resume --experiment ../$(OUT) --device $(DEVICE)
	$(AE) predict --experiment ../$(OUT) --device $(DEVICE)

ae-bench:
	$(AE) bench --experiment ../$(OUT) --device $(DEVICE)

evaluate:
	$(GO) run ./cmd/evaluate -experiment $(OUT) -reference $(REF)

# Each stage consumes the previous stage's files, even under make -j.
pipeline:
	$(MAKE) data EXP=$(EXP)
	$(MAKE) markov EXP=$(EXP)
	$(MAKE) recover EXP=$(EXP)
	$(MAKE) ae EXP=$(EXP)
	$(MAKE) evaluate EXP=$(EXP)

smoke:
	$(MAKE) pipeline EXP=smoke

figures:
	cd autoencoder && $(UV) run --locked --group docs python ../scripts/render_figures.py

analysis:
	cd autoencoder && $(UV) run --locked python ../scripts/summarize_results.py

verify-results:
	cd autoencoder && $(UV) run --locked python ../scripts/verify_results.py
