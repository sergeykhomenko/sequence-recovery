package restore

import (
	"math/rand/v2"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/markov"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

var alphabet = []string{"A", "B", "C", "D", "E"}

func models(t *testing.T, maxOrder int, seqs ...[]string) []*markov.Model {
	t.Helper()
	tr, err := markov.NewTrainer(maxOrder)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range seqs {
		tr.Add(s)
	}
	paths, err := tr.Export(filepath.Join(t.TempDir(), "m"))
	if err != nil {
		t.Fatal(err)
	}
	var out []*markov.Model
	for _, p := range paths {
		m, err := markov.LoadModel(p)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, m)
	}
	return out
}

// trained learns 5x "A B C D" and 1x "A C B D": after A comes B (5) or C (1),
// after B comes C (5) or D (1), after the pair A B always C.
func trained(t *testing.T) *Markov {
	var seqs [][]string
	for range 5 {
		seqs = append(seqs, []string{"A", "B", "C", "D"})
	}
	seqs = append(seqs, []string{"A", "C", "B", "D"})
	m, err := NewMarkov(models(t, 2, seqs...))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// challenge builds a challenge from tokens, "" marking a mask.
func challenge(tokens ...string) task.Challenge {
	return task.Challenge{ID: "c", Window: task.Window(tokens)}
}

func TestMarkovFill(t *testing.T) {
	m := trained(t)
	cases := []struct {
		name   string
		window []string
		order  int
		fill   []string
		used   []int
	}{
		{"most frequent successor", []string{"A", "", "", "D"}, 1, []string{"B", "C"}, []int{1, 1}},
		{"skips used symbols", []string{"A", "", "B", "D"}, 1, []string{"C"}, []int{1}},
		{"full order when known", []string{"A", "B", "", "D"}, 2, []string{"C"}, []int{2}},
		{"backs off on unknown context", []string{"D", "A", "", "E"}, 2, []string{"B"}, []int{1}},
	}
	for _, tc := range cases {
		p, err := m.Fill(challenge(tc.window...), alphabet, tc.order, rand.New(rand.NewPCG(1, 1)))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !reflect.DeepEqual(p.Fill, tc.fill) || !reflect.DeepEqual(p.OrdersUsed, tc.used) {
			t.Errorf("%s: fill %v from orders %v, want %v from %v", tc.name, p.Fill, p.OrdersUsed, tc.fill, tc.used)
		}
	}
}

func TestMarkovFallsBackToRandomAtWindowStart(t *testing.T) {
	p, err := trained(t).Fill(challenge("", "B", "C", "D"), alphabet, 2, rand.New(rand.NewPCG(1, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if p.OrdersUsed[0] != 0 || (p.Fill[0] != "A" && p.Fill[0] != "E") {
		t.Errorf("fill %v from orders %v, want a random unused symbol (order 0)", p.Fill, p.OrdersUsed)
	}
}

func TestNewMarkovRejectsMissingOrders(t *testing.T) {
	ms := models(t, 2, []string{"A", "B"})
	if _, err := NewMarkov(ms[1:]); err == nil {
		t.Error("accepted models starting at order 2")
	}
	if _, err := NewMarkov(nil); err == nil {
		t.Error("accepted no models")
	}
}

func TestRandomFillsDistinctUnusedSymbols(t *testing.T) {
	for s := range uint64(20) {
		p, err := Random(challenge("A", "", "", "D"), alphabet, rand.New(rand.NewPCG(s, s)))
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Fill) != 2 || p.Fill[0] == p.Fill[1] {
			t.Fatalf("fill %v", p.Fill)
		}
		for _, tok := range p.Fill {
			if tok != "B" && tok != "C" && tok != "E" {
				t.Fatalf("fill %v uses a kept symbol", p.Fill)
			}
		}
	}
}
