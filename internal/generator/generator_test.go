package generator

import (
	"math/rand/v2"
	"reflect"
	"testing"
)

func testConfig() Config {
	return Config{Size: 60, Entries: 5, MinPrereq: 1, MaxPrereq: 3, Decay: 0.8, Seed: 42}
}

func TestGenerateShape(t *testing.T) {
	cfg := testConfig()
	g, err := Generate(cfg, DecayStrategy{Decay: cfg.Decay})
	if err != nil {
		t.Fatal(err)
	}

	if len(g.Nodes) != cfg.Size {
		t.Fatalf("nodes = %d, want %d", len(g.Nodes), cfg.Size)
	}

	pos := map[string]int{}
	for i, n := range g.Nodes {
		pos[n.ID] = i
		if n.Entry != (i < cfg.Entries) {
			t.Errorf("node %s entry = %v", n.ID, n.Entry)
		}
	}

	seen := map[[2]string]bool{}
	for _, e := range g.Edges {
		if pos[e.From] >= pos[e.To] {
			t.Errorf("edge %s->%s goes backwards", e.From, e.To)
		}
		key := [2]string{e.From, e.To}
		if seen[key] {
			t.Errorf("duplicate edge %s->%s", e.From, e.To)
		}
		seen[key] = true
	}

	in := g.InDegree()
	for i, n := range g.Nodes {
		d := in[n.ID]
		if i < cfg.Entries {
			if d != 0 {
				t.Errorf("entry %s has %d prerequisites", n.ID, d)
			}
			continue
		}
		if d < cfg.MinPrereq || d > cfg.MaxPrereq {
			t.Errorf("node %s has %d prerequisites, want [%d, %d]", n.ID, d, cfg.MinPrereq, cfg.MaxPrereq)
		}
	}
}

func TestGenerateCapsPrereqByAvailableNodes(t *testing.T) {
	cfg := Config{Size: 5, Entries: 1, MinPrereq: 3, MaxPrereq: 3, Decay: 1, Seed: 1}
	g, err := Generate(cfg, DecayStrategy{Decay: cfg.Decay})
	if err != nil {
		t.Fatal(err)
	}
	in := g.InDegree()
	// B has only A before it, C has A and B.
	if in["B"] != 1 || in["C"] != 2 || in["D"] != 3 || in["E"] != 3 {
		t.Errorf("in-degrees = %v", in)
	}
}

func TestGenerateDeterministic(t *testing.T) {
	cfg := testConfig()
	a, _ := Generate(cfg, DecayStrategy{Decay: cfg.Decay})
	b, _ := Generate(cfg, DecayStrategy{Decay: cfg.Decay})
	if !reflect.DeepEqual(a, b) {
		t.Error("same seed produced different graphs")
	}

	cfg.Seed++
	c, _ := Generate(cfg, DecayStrategy{Decay: cfg.Decay})
	if reflect.DeepEqual(a.Edges, c.Edges) {
		t.Error("different seeds produced identical edges")
	}
}

func TestValidate(t *testing.T) {
	bad := []Config{
		{Size: 10, Entries: 0, MinPrereq: 1, MaxPrereq: 2, Decay: 0.5},
		{Size: 3, Entries: 5, MinPrereq: 1, MaxPrereq: 2, Decay: 0.5},
		{Size: 10, Entries: 5, MinPrereq: 0, MaxPrereq: 2, Decay: 0.5},
		{Size: 10, Entries: 5, MinPrereq: 3, MaxPrereq: 2, Decay: 0.5},
		{Size: 10, Entries: 5, MinPrereq: 1, MaxPrereq: 2, Decay: 0},
		{Size: 10, Entries: 5, MinPrereq: 1, MaxPrereq: 2, Decay: 1.5},
	}
	for _, cfg := range bad {
		if err := cfg.Validate(); err == nil {
			t.Errorf("Validate(%+v) = nil, want error", cfg)
		}
	}
}

func TestDecayStrategyPrefersRecent(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 7))
	s := DecayStrategy{Decay: 0.5}
	const i, trials = 20, 2000
	counts := make([]int, i)
	for range trials {
		for _, j := range s.Pick(rng, i, 1) {
			counts[j]++
		}
	}
	if counts[i-1] <= counts[0] {
		t.Errorf("most recent picked %d times, oldest %d times", counts[i-1], counts[0])
	}
}
