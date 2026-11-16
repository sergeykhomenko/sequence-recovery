package graph

import (
	"errors"
	"fmt"
)

// Closure answers "is a a transitive prerequisite of b" with one bit lookup.
type Closure struct {
	index map[string]int
	anc   [][]uint64 // anc[v] has bit u set when u is an ancestor of v
}

func NewClosure(g *Graph) (*Closure, error) {
	n := len(g.Nodes)
	c := &Closure{index: make(map[string]int, n), anc: make([][]uint64, n)}
	for i, node := range g.Nodes {
		if _, dup := c.index[node.ID]; dup {
			return nil, fmt.Errorf("duplicate node %q", node.ID)
		}
		c.index[node.ID] = i
	}
	prereqs := make([][]int, n)
	dependants := make([][]int, n)
	left := make([]int, n)
	for _, e := range g.Edges {
		from, ok1 := c.index[e.From]
		to, ok2 := c.index[e.To]
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("edge %s->%s has an unknown node", e.From, e.To)
		}
		prereqs[to] = append(prereqs[to], from)
		dependants[from] = append(dependants[from], to)
		left[to]++
	}

	// Kahn's order visits every node after all of its prerequisites.
	words := (n + 63) / 64
	var queue []int
	for v := range n {
		if left[v] == 0 {
			queue = append(queue, v)
		}
	}
	done := 0
	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		done++
		c.anc[v] = make([]uint64, words)
		for _, u := range prereqs[v] {
			c.anc[v][u/64] |= 1 << (u % 64)
			for k := range words {
				c.anc[v][k] |= c.anc[u][k]
			}
		}
		for _, w := range dependants[v] {
			if left[w]--; left[w] == 0 {
				queue = append(queue, w)
			}
		}
	}
	if done != n {
		return nil, errors.New("graph has a cycle")
	}
	return c, nil
}

func (c *Closure) Has(id string) bool {
	_, ok := c.index[id]
	return ok
}

// Precedes reports whether a is a transitive prerequisite of b.
func (c *Closure) Precedes(a, b string) bool {
	ia, ok1 := c.index[a]
	ib, ok2 := c.index[b]
	if !ok1 || !ok2 {
		return false
	}
	return c.anc[ib][ia/64]&(1<<(ia%64)) != 0
}
