// Package graph holds the synthetic knowledge graph model and its JSON form.
package graph

import (
	"encoding/json"
	"io"
)

type Node struct {
	ID    string `json:"id"`
	Entry bool   `json:"entry"`
}

// Edge means From is a prerequisite of To.
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type Graph struct {
	Params any    `json:"params,omitempty"`
	Nodes  []Node `json:"nodes"`
	Edges  []Edge `json:"edges"`
}

// InDegree returns the number of prerequisites per node ID.
func (g *Graph) InDegree() map[string]int {
	deg := make(map[string]int, len(g.Nodes))
	for _, e := range g.Edges {
		deg[e.To]++
	}
	return deg
}

// OutDegree returns the number of dependants per node ID.
func (g *Graph) OutDegree() map[string]int {
	deg := make(map[string]int, len(g.Nodes))
	for _, e := range g.Edges {
		deg[e.From]++
	}
	return deg
}

// ReadJSON decodes a graph in the form written by WriteJSON.
func ReadJSON(r io.Reader) (*Graph, error) {
	var g Graph
	if err := json.NewDecoder(r).Decode(&g); err != nil {
		return nil, err
	}
	return &g, nil
}

func (g *Graph) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(g)
}
