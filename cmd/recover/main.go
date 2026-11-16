// Command recover fills the masks of every challenge of an experiment with the
// random baseline or with Markov chains of orders 1..max-order.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/sergeykhomenko/sequence-recovery/internal/jsonl"
	"github.com/sergeykhomenko/sequence-recovery/internal/markov"
	"github.com/sergeykhomenko/sequence-recovery/internal/report"
	"github.com/sergeykhomenko/sequence-recovery/internal/restore"
	"github.com/sergeykhomenko/sequence-recovery/internal/seed"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

func main() {
	dir := flag.String("experiment", "", "experiment directory")
	method := flag.String("method", "random", "random or markov")
	maxOrder := flag.Int("max-order", 5, "markov: predict with every order 1..max-order")
	flag.Parse()
	if *dir == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(store.Store{Dir: *dir}, *method, *maxOrder); err != nil {
		log.Fatal(err)
	}
}

func run(st store.Store, method string, maxOrder int) error {
	cfg, err := st.ReadConfig()
	if err != nil {
		return err
	}
	challenges, err := jsonl.Read[task.Challenge](st.ChallengesPath())
	if err != nil {
		return err
	}
	if err := os.MkdirAll(st.PredictionsDir(), 0o755); err != nil {
		return err
	}
	alphabets := map[string][]string{}
	alphabet := func(id string) ([]string, error) {
		if a, ok := alphabets[id]; ok {
			return a, nil
		}
		g, _, err := st.ReadGraph(id)
		if err != nil {
			return nil, err
		}
		a := make([]string, len(g.Nodes))
		for i, n := range g.Nodes {
			a[i] = n.ID
		}
		alphabets[id] = a
		return a, nil
	}

	switch method {
	case "random":
		preds := make([]task.Prediction, 0, len(challenges))
		for _, c := range challenges {
			a, err := alphabet(c.Graph)
			if err != nil {
				return err
			}
			p, err := restore.Random(c, a, seed.RNG(cfg.Seed, "random", c.ID))
			if err != nil {
				return err
			}
			preds = append(preds, p)
		}
		return jsonl.Write(st.PredictionPath("random"), preds)

	case "markov":
		for _, k := range cfg.Orders.MarkovBudgets {
			preds := make([][]task.Prediction, maxOrder)
			var chains *restore.Markov
			loaded := ""
			for _, c := range challenges {
				if c.Graph != loaded {
					if chains, err = loadChains(st, k, c.Graph, maxOrder); err != nil {
						return err
					}
					loaded = c.Graph
				}
				a, err := alphabet(c.Graph)
				if err != nil {
					return err
				}
				for n := 1; n <= maxOrder; n++ {
					p, err := chains.Fill(c, a, n, seed.RNG(cfg.Seed, report.MarkovMethod(n, k), c.ID))
					if err != nil {
						return err
					}
					preds[n-1] = append(preds[n-1], p)
				}
			}
			for n, ps := range preds {
				if err := jsonl.Write(st.PredictionPath(report.MarkovMethod(n+1, k)), ps); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return fmt.Errorf("unknown method %q (use random or markov)", method)
}

func loadChains(st store.Store, budget int, graphID string, maxOrder int) (*restore.Markov, error) {
	models := make([]*markov.Model, maxOrder)
	for k := 1; k <= maxOrder; k++ {
		m, err := markov.LoadModel(fmt.Sprintf("%s_%d.json", st.MarkovPrefix(budget, graphID), k))
		if err != nil {
			return nil, err
		}
		models[k-1] = m
	}
	return restore.NewMarkov(models)
}
