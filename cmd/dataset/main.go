// Command dataset builds an experiment directory from a config: graphs,
// trajectories, test challenges and autoencoder training windows.
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/sergeykhomenko/sequence-recovery/internal/dataset"
	"github.com/sergeykhomenko/sequence-recovery/internal/experiment"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
)

func main() {
	configPath := flag.String("config", "experiments/base.json", "experiment config")
	out := flag.String("out", "", "experiment directory; defaults to out/<name>")
	flag.Parse()

	c, err := experiment.Load(*configPath)
	if err != nil {
		log.Fatal(err)
	}
	dir := *out
	if dir == "" {
		dir = filepath.Join("out", c.Name)
	}
	if err := dataset.Build(c, store.Store{Dir: dir}); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("built experiment %q in %s: %d train, %d val, %d test graphs\n",
		c.Name, dir, c.Graphs.Train, c.Graphs.Val, c.Graphs.Test)
}
