// Package restore fills the masks of a challenge: the random baseline and
// Markov chains with backoff. Both follow the experiment rules: only masks
// are filled, a symbol appears in a window at most once, no abstentions.
package restore

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/sergeykhomenko/sequence-recovery/internal/markov"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

// Allowed lists the alphabet symbols that are not in the window yet, in alphabet order.
func Allowed(alphabet []string, w task.Window) []string {
	used := make(map[string]bool, len(w))
	for _, t := range w {
		if t != "" {
			used[t] = true
		}
	}
	out := make([]string, 0, len(alphabet))
	for _, s := range alphabet {
		if !used[s] {
			out = append(out, s)
		}
	}
	return out
}

// Random fills the masks with distinct random allowed symbols.
func Random(c task.Challenge, alphabet []string, rng *rand.Rand) (task.Prediction, error) {
	pool := Allowed(alphabet, c.Window)
	m := len(c.Window.MaskPositions())
	if len(pool) < m {
		return task.Prediction{}, fmt.Errorf("%s: %d masks but %d unused symbols", c.ID, m, len(pool))
	}
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	return task.Prediction{ID: c.ID, Fill: pool[:m]}, nil
}

// Markov fills masks left to right with chains of orders 1..len(models).
type Markov struct {
	models []*markov.Model // models[k-1] has order k
}

func NewMarkov(models []*markov.Model) (*Markov, error) {
	if len(models) == 0 {
		return nil, errors.New("no models")
	}
	for i, m := range models {
		if m.Order != i+1 {
			return nil, fmt.Errorf("model %d has order %d, want %d", i, m.Order, i+1)
		}
	}
	return &Markov{models: models}, nil
}

// Fill predicts with order n: each mask gets the most frequent allowed
// successor of the last n tokens, backing off to shorter contexts when the
// window is too short on the left, the context is unknown or all of its
// successors are used, and to a random allowed symbol as the last resort.
func (m *Markov) Fill(c task.Challenge, alphabet []string, n int, rng *rand.Rand) (task.Prediction, error) {
	if n < 1 || n > len(m.models) {
		return task.Prediction{}, fmt.Errorf("order %d is outside 1..%d", n, len(m.models))
	}
	w := slices.Clone(c.Window)
	used := make(map[string]bool, len(w))
	for _, t := range w {
		if t != "" {
			used[t] = true
		}
	}
	pred := task.Prediction{ID: c.ID}
	for _, i := range c.Window.MaskPositions() {
		tok, order := "", 0
		for k := min(n, i); k >= 1; k-- {
			if t := best(m.models[k-1].Next(w[i-k:i]), used, rng); t != "" {
				tok, order = t, k
				break
			}
		}
		if tok == "" {
			pool := Allowed(alphabet, w)
			if len(pool) == 0 {
				return task.Prediction{}, fmt.Errorf("%s: no symbols left for position %d", c.ID, i)
			}
			tok = pool[rng.IntN(len(pool))]
		}
		w[i] = tok
		used[tok] = true
		pred.Fill = append(pred.Fill, tok)
		pred.OrdersUsed = append(pred.OrdersUsed, order)
	}
	return pred, nil
}

// best returns the most frequent unused successor, breaking ties at random.
func best(next map[string]int, used map[string]bool, rng *rand.Rand) string {
	var top []string
	most := 0
	for tok, n := range next {
		if used[tok] || tok == markov.EndToken || tok == markov.StartToken {
			continue
		}
		switch {
		case n > most:
			most, top = n, []string{tok}
		case n == most:
			top = append(top, tok)
		}
	}
	if len(top) == 0 {
		return ""
	}
	slices.Sort(top) // map order is random; sort before the seeded draw
	return top[rng.IntN(len(top))]
}
