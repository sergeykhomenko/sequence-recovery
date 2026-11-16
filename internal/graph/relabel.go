package graph

import (
	"fmt"
	"math/rand/v2"
)

// Relabel returns a copy of g with every node ID replaced by mapping[ID].
func (g *Graph) Relabel(mapping map[string]string) (*Graph, error) {
	out := &Graph{Params: g.Params, Nodes: make([]Node, len(g.Nodes)), Edges: make([]Edge, len(g.Edges))}
	used := make(map[string]bool, len(g.Nodes))
	for i, n := range g.Nodes {
		name, ok := mapping[n.ID]
		if !ok {
			return nil, fmt.Errorf("no new name for node %q", n.ID)
		}
		if used[name] {
			return nil, fmt.Errorf("name %q is given to two nodes", name)
		}
		used[name] = true
		out.Nodes[i] = Node{ID: name, Entry: n.Entry}
	}
	for i, e := range g.Edges {
		from, ok1 := mapping[e.From]
		to, ok2 := mapping[e.To]
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("edge %s->%s has an unknown node", e.From, e.To)
		}
		out.Edges[i] = Edge{From: from, To: to}
	}
	return out, nil
}

// Shuffle puts nodes and edges in random order, so positions in the file
// carry no information about how the graph was generated.
func (g *Graph) Shuffle(rng *rand.Rand) {
	rng.Shuffle(len(g.Nodes), func(i, j int) { g.Nodes[i], g.Nodes[j] = g.Nodes[j], g.Nodes[i] })
	rng.Shuffle(len(g.Edges), func(i, j int) { g.Edges[i], g.Edges[j] = g.Edges[j], g.Edges[i] })
}
