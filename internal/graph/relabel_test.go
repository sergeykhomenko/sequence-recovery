package graph

import (
	"fmt"
	"math/rand/v2"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestRelabel(t *testing.T) {
	g := &Graph{
		Nodes: []Node{{ID: "A", Entry: true}, {ID: "B"}},
		Edges: []Edge{{From: "A", To: "B"}},
	}
	got, err := g.Relabel(map[string]string{"A": "X1", "B": "Q7"})
	if err != nil {
		t.Fatal(err)
	}
	if want := []Node{{ID: "X1", Entry: true}, {ID: "Q7"}}; !reflect.DeepEqual(got.Nodes, want) {
		t.Errorf("nodes = %v, want %v", got.Nodes, want)
	}
	if want := []Edge{{From: "X1", To: "Q7"}}; !reflect.DeepEqual(got.Edges, want) {
		t.Errorf("edges = %v, want %v", got.Edges, want)
	}
	if g.Nodes[0].ID != "A" {
		t.Error("Relabel changed the original graph")
	}
}

func TestRelabelRejectsBadMappings(t *testing.T) {
	g := &Graph{Nodes: []Node{{ID: "A"}, {ID: "B"}}, Edges: []Edge{{From: "A", To: "B"}}}
	for name, mapping := range map[string]map[string]string{
		"missing node":    {"A": "X1"},
		"same name twice": {"A": "X1", "B": "X1"},
	} {
		if _, err := g.Relabel(mapping); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestShuffleKeepsContent(t *testing.T) {
	g := &Graph{}
	for i := range 20 {
		g.Nodes = append(g.Nodes, Node{ID: fmt.Sprintf("N%02d", i)})
		if i > 0 {
			g.Edges = append(g.Edges, Edge{From: fmt.Sprintf("N%02d", i-1), To: fmt.Sprintf("N%02d", i)})
		}
	}
	nodes, edges := slices.Clone(g.Nodes), slices.Clone(g.Edges)

	g.Shuffle(rand.New(rand.NewPCG(1, 1)))

	if reflect.DeepEqual(g.Nodes, nodes) {
		t.Error("nodes kept their order")
	}
	gotNodes := slices.Clone(g.Nodes)
	slices.SortFunc(gotNodes, func(a, b Node) int { return strings.Compare(a.ID, b.ID) })
	gotEdges := slices.Clone(g.Edges)
	slices.SortFunc(gotEdges, func(a, b Edge) int { return strings.Compare(a.From, b.From) })
	if !reflect.DeepEqual(gotNodes, nodes) || !reflect.DeepEqual(gotEdges, edges) {
		t.Error("shuffle changed the content of the graph")
	}
}
