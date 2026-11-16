package sampler

import (
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/generator"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

func nodes(ids ...string) []graph.Node {
	out := make([]graph.Node, len(ids))
	for i, id := range ids {
		out[i] = graph.Node{ID: id}
	}
	return out
}

func TestSampleProducesExactlyValidOrders(t *testing.T) {
	// A->B, B->C, A->D, C->D fixes the order A B C D; E is independent.
	g := &graph.Graph{
		Nodes: nodes("A", "B", "C", "D", "E"),
		Edges: []graph.Edge{{From: "A", To: "B"}, {From: "B", To: "C"}, {From: "A", To: "D"}, {From: "C", To: "D"}},
	}
	s, err := New(g)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(1, 1))
	seen := map[string]bool{}
	for range 500 {
		seen[strings.Join(s.Sample(rng), " ")] = true
	}

	want := []string{
		"E A B C D",
		"A E B C D",
		"A B E C D",
		"A B C E D",
		"A B C D E",
	}
	var got []string
	for order := range seen {
		got = append(got, order)
	}
	slices.Sort(got)
	slices.Sort(want)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("orders = %q, want %q", got, want)
	}
}

func TestSampleOnGeneratedGraph(t *testing.T) {
	cfg := generator.Config{Size: 300, Entries: 10, MinPrereq: 1, MaxPrereq: 8, Decay: 0.8, Seed: 3}
	g, err := generator.Generate(cfg, generator.DecayStrategy{Decay: cfg.Decay})
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(g)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(2, 2))
	for range 50 {
		order := s.Sample(rng)
		if len(order) != cfg.Size {
			t.Fatalf("order has %d nodes, want %d", len(order), cfg.Size)
		}
		pos := make(map[string]int, len(order))
		for i, id := range order {
			if _, dup := pos[id]; dup {
				t.Fatalf("node %s appears twice", id)
			}
			pos[id] = i
		}
		for _, e := range g.Edges {
			if pos[e.From] > pos[e.To] {
				t.Fatalf("%s comes before its prerequisite %s", e.To, e.From)
			}
		}
	}
}

func TestSampleDeterministicForSeed(t *testing.T) {
	g := &graph.Graph{Nodes: nodes("A", "B", "C", "D", "E", "F")}
	s, err := New(g)
	if err != nil {
		t.Fatal(err)
	}

	a := rand.New(rand.NewPCG(5, 5))
	b := rand.New(rand.NewPCG(5, 5))
	for range 20 {
		if x, y := s.Sample(a), s.Sample(b); !reflect.DeepEqual(x, y) {
			t.Fatalf("same seed gave %v and %v", x, y)
		}
	}
}

func TestNewRejectsInvalidGraphs(t *testing.T) {
	cases := map[string]*graph.Graph{
		"cycle": {
			Nodes: nodes("A", "B"),
			Edges: []graph.Edge{{From: "A", To: "B"}, {From: "B", To: "A"}},
		},
		"self loop": {
			Nodes: nodes("A"),
			Edges: []graph.Edge{{From: "A", To: "A"}},
		},
		"unknown from": {
			Nodes: nodes("A"),
			Edges: []graph.Edge{{From: "X", To: "A"}},
		},
		"unknown to": {
			Nodes: nodes("A"),
			Edges: []graph.Edge{{From: "A", To: "X"}},
		},
		"duplicate node": {
			Nodes: nodes("A", "A"),
		},
		"empty": {},
	}
	for name, g := range cases {
		if _, err := New(g); err == nil {
			t.Errorf("%s: New returned no error", name)
		}
	}
}
