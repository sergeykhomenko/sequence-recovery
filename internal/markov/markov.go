// Package markov trains gomarkov chains of orders 1..N on the same sequences
// and stores chain n as <prefix>_<n>.json, the layout sequences-recovery loads.
package markov

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mb-14/gomarkov"
)

// Padding tokens gomarkov wraps every trained sequence in.
const (
	StartToken = gomarkov.StartToken
	EndToken   = gomarkov.EndToken
)

// Trainer feeds every sequence into the chains of all orders at once.
type Trainer struct {
	chains []*gomarkov.Chain
}

func NewTrainer(maxOrder int) (*Trainer, error) {
	if maxOrder < 1 {
		return nil, errors.New("max order must be >= 1")
	}
	chains := make([]*gomarkov.Chain, maxOrder)
	for i := range chains {
		chains[i] = gomarkov.NewChain(i + 1)
	}
	return &Trainer{chains: chains}, nil
}

func (t *Trainer) Add(seq []string) {
	for _, c := range t.chains {
		c.Add(seq)
	}
}

// Export writes chain n to <prefix>_<n>.json and returns the written paths.
func (t *Trainer) Export(prefix string) ([]string, error) {
	paths := make([]string, 0, len(t.chains))
	for _, c := range t.chains {
		data, err := json.Marshal(c)
		if err != nil {
			return nil, fmt.Errorf("marshal order %d: %w", c.Order, err)
		}
		path := fmt.Sprintf("%s_%d.json", prefix, c.Order)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, nil
}
