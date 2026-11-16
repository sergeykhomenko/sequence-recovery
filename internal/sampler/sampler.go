// Package sampler draws random topological orders of a graph.Graph: learning
// trajectories that pass every node once, each after all of its prerequisites.
package sampler

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

type Sampler struct {
	ids        []string
	dependants [][]int // dependants[i] lists the nodes that have i as a prerequisite
	prereqs    []int   // number of prerequisites per node
}

// New indexes the graph and checks that every order can include all nodes:
// node IDs are unique, edges reference known nodes and there are no cycles.
func New(g *graph.Graph) (*Sampler, error) {
	if len(g.Nodes) == 0 {
		return nil, errors.New("graph has no nodes")
	}

	s := &Sampler{
		ids:        make([]string, len(g.Nodes)),
		dependants: make([][]int, len(g.Nodes)),
		prereqs:    make([]int, len(g.Nodes)),
	}
	index := make(map[string]int, len(g.Nodes))
	for i, n := range g.Nodes {
		if _, dup := index[n.ID]; dup {
			return nil, fmt.Errorf("duplicate node %q", n.ID)
		}
		index[n.ID] = i
		s.ids[i] = n.ID
	}

	for _, e := range g.Edges {
		from, ok := index[e.From]
		if !ok {
			return nil, fmt.Errorf("edge %s->%s: unknown node %q", e.From, e.To, e.From)
		}
		to, ok := index[e.To]
		if !ok {
			return nil, fmt.Errorf("edge %s->%s: unknown node %q", e.From, e.To, e.To)
		}
		s.dependants[from] = append(s.dependants[from], to)
		s.prereqs[to]++
	}

	// Any order drops the nodes on a cycle, so one pass tells whether the graph is a DAG.
	if n := len(s.Sample(rand.New(rand.NewPCG(0, 0)))); n != len(s.ids) {
		return nil, fmt.Errorf("graph has a cycle: only %d of %d nodes can be ordered", n, len(s.ids))
	}
	return s, nil
}

// Sample returns one random topological order (Kahn's algorithm): the next
// node is chosen uniformly among the nodes whose prerequisites are all done.
func (s *Sampler) Sample(rng *rand.Rand) []string {
	left := slices.Clone(s.prereqs)
	var ready []int
	for i, n := range left {
		if n == 0 {
			ready = append(ready, i)
		}
	}

	order := make([]string, 0, len(s.ids))
	for len(ready) > 0 {
		k := rng.IntN(len(ready))
		v := ready[k]
		ready[k] = ready[len(ready)-1]
		ready = ready[:len(ready)-1]

		order = append(order, s.ids[v])
		for _, w := range s.dependants[v] {
			left[w]--
			if left[w] == 0 {
				ready = append(ready, w)
			}
		}
	}
	return order
}
