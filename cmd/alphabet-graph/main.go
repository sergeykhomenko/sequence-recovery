// Command alphabet-graph generates a synthetic knowledge graph over an alphabet
// of the given size and writes it as JSON plus an image.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/sergeykhomenko/sequence-recovery/internal/generator"
	"github.com/sergeykhomenko/sequence-recovery/internal/render"
)

func main() {
	var cfg generator.Config
	flag.IntVar(&cfg.Size, "size", 0, "alphabet size (number of nodes), required")
	flag.IntVar(&cfg.Entries, "entries", 5, "number of entry points (nodes without prerequisites)")
	flag.IntVar(&cfg.MinPrereq, "min-prereq", 1, "minimum prerequisites per non-entry node")
	flag.IntVar(&cfg.MaxPrereq, "max-prereq", 3, "maximum prerequisites per non-entry node")
	flag.Float64Var(&cfg.Decay, "decay", 0.8, "locality in (0, 1]: weight of a candidate is decay^distance; 1 = uniform")
	flag.Uint64Var(&cfg.Seed, "seed", 1, "random seed")
	out := flag.String("out", "out/graph", "output path prefix; writes <out>.json and <out>.<format>")
	formatName := flag.String("format", "png", "image format: png or svg")
	flag.Parse()

	if cfg.Size == 0 {
		flag.Usage()
		os.Exit(2)
	}

	format, err := render.ParseFormat(*formatName)
	if err != nil {
		log.Fatal(err)
	}

	g, err := generator.Generate(cfg, generator.DecayStrategy{Decay: cfg.Decay})
	if err != nil {
		log.Fatal(err)
	}

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		log.Fatal(err)
	}

	jsonPath := *out + ".json"
	f, err := os.Create(jsonPath)
	if err != nil {
		log.Fatal(err)
	}
	if err := g.WriteJSON(f); err != nil {
		f.Close()
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}

	imgPath := *out + "." + *formatName
	if err := render.Render(context.Background(), g, format, imgPath); err != nil {
		log.Fatal(err)
	}

	isolated := 0
	outDeg := g.OutDegree()
	for _, n := range g.Nodes {
		if n.Entry && outDeg[n.ID] == 0 {
			isolated++
		}
	}
	fmt.Printf("nodes: %d, edges: %d, isolated entries: %d\n", len(g.Nodes), len(g.Edges), isolated)
	fmt.Printf("wrote %s, %s\n", jsonPath, imgPath)
}
