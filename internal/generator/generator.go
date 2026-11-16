// Package generator builds a synthetic knowledge graph over an alphabet:
// the first Entries letters are entry points, every next letter gets
// prerequisites among the letters added before it.
package generator

import (
	"errors"
	"math/rand/v2"

	"github.com/sergeykhomenko/sequence-recovery/internal/alphabet"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

type Config struct {
	Size      int     `json:"size"`
	Entries   int     `json:"entries"`
	MinPrereq int     `json:"min_prereq"`
	MaxPrereq int     `json:"max_prereq"`
	Decay     float64 `json:"decay"`
	Seed      uint64  `json:"seed"`
}

func (c Config) Validate() error {
	switch {
	case c.Entries < 1:
		return errors.New("entries must be >= 1")
	case c.Size < c.Entries:
		return errors.New("size must be >= entries")
	case c.MinPrereq < 1:
		return errors.New("min-prereq must be >= 1")
	case c.MinPrereq > c.MaxPrereq:
		return errors.New("min-prereq must be <= max-prereq")
	case c.Decay <= 0 || c.Decay > 1:
		return errors.New("decay must be in (0, 1]")
	}
	return nil
}

// Generate builds the graph. Edges always go from an earlier letter to a
// later one, so the result is a DAG by construction.
func Generate(cfg Config, s Strategy) (*graph.Graph, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed))
	letters := alphabet.Letters(cfg.Size)
	g := &graph.Graph{Params: cfg}

	for i, id := range letters {
		entry := i < cfg.Entries
		g.Nodes = append(g.Nodes, graph.Node{ID: id, Entry: entry})
		if entry {
			continue
		}

		hi := min(cfg.MaxPrereq, i)
		lo := min(cfg.MinPrereq, hi)
		k := lo + rng.IntN(hi-lo+1)
		for _, j := range s.Pick(rng, i, k) {
			g.Edges = append(g.Edges, graph.Edge{From: letters[j], To: id})
		}
	}
	return g, nil
}
