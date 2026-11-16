// Command markov-train samples random topological orders (learning trajectories)
// from a graph written by alphabet-graph and trains gomarkov chains of orders
// 1..max-order on them.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
	"github.com/sergeykhomenko/sequence-recovery/internal/markov"
	"github.com/sergeykhomenko/sequence-recovery/internal/sampler"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
)

func main() {
	graphPath := flag.String("graph", "out/graph.json", "graph JSON written by alphabet-graph")
	samples := flag.Int("samples", 1000, "number of sampled sequences to train on")
	seed := flag.Uint64("seed", 1, "random seed")
	maxOrder := flag.Int("max-order", 5, "train chains of orders 1..max-order")
	out := flag.String("out", "out/markov", "output path prefix; writes <out>_<order>.json")
	seqPath := flag.String("sequences", "", "optional file for the sampled sequences: one per line, tokens separated by spaces")
	experimentDir := flag.String("experiment", "", "experiment directory: train chains for every test graph from orders/<id>.txt into markov/<id>_<order>.json")
	flag.Parse()

	if *experimentDir != "" {
		if err := trainExperiment(store.Store{Dir: *experimentDir}, *maxOrder); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *samples < 1 {
		log.Fatal("samples must be >= 1")
	}

	g, err := readGraph(*graphPath)
	if err != nil {
		log.Fatal(err)
	}
	s, err := sampler.New(g)
	if err != nil {
		log.Fatal(err)
	}
	tr, err := markov.NewTrainer(*maxOrder)
	if err != nil {
		log.Fatal(err)
	}

	var seqFile *os.File
	var seqOut *bufio.Writer
	if *seqPath != "" {
		if seqFile, err = create(*seqPath); err != nil {
			log.Fatal(err)
		}
		seqOut = bufio.NewWriter(seqFile)
	}

	rng := rand.New(rand.NewPCG(*seed, *seed))
	for range *samples {
		seq := s.Sample(rng)
		tr.Add(seq)
		if seqOut != nil {
			fmt.Fprintln(seqOut, strings.Join(seq, " "))
		}
	}

	if seqOut != nil {
		if err := seqOut.Flush(); err != nil {
			log.Fatal(err)
		}
		if err := seqFile.Close(); err != nil {
			log.Fatal(err)
		}
	}

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		log.Fatal(err)
	}
	paths, err := tr.Export(*out)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("sampled %d sequences over %d nodes\n", *samples, len(g.Nodes))
	fmt.Printf("wrote %s\n", strings.Join(paths, ", "))
	if *seqPath != "" {
		fmt.Printf("wrote %s\n", *seqPath)
	}
}

func readGraph(path string) (*graph.Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return graph.ReadJSON(f)
}

// create makes the parent directory and then the file.
func create(path string) (*os.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.Create(path)
}

// trainExperiment trains, for every test graph and every budget K of the
// config, chains on the first K stored trajectories.
func trainExperiment(st store.Store, maxOrder int) error {
	cfg, err := st.ReadConfig()
	if err != nil {
		return err
	}
	ids, err := st.GraphIDs(store.Test)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("%s: no test graphs", st.Dir)
	}
	budgets := cfg.Orders.MarkovBudgets
	for _, k := range budgets {
		if err := os.MkdirAll(filepath.Dir(st.MarkovPrefix(k, ids[0])), 0o755); err != nil {
			return err
		}
	}
	for _, id := range ids {
		orders, err := st.ReadOrders(id)
		if err != nil {
			return err
		}
		for _, k := range budgets {
			if k > len(orders) {
				return fmt.Errorf("%s: budget %d but only %d trajectories", id, k, len(orders))
			}
			tr, err := markov.NewTrainer(maxOrder)
			if err != nil {
				return err
			}
			for _, o := range orders[:k] {
				tr.Add(o)
			}
			if _, err := tr.Export(st.MarkovPrefix(k, id)); err != nil {
				return err
			}
		}
		fmt.Printf("%s: orders 1..%d, budgets %v\n", id, maxOrder, budgets)
	}
	return nil
}
