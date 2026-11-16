// Package render draws a graph.Graph with go-graphviz (embedded dot, no system install).
package render

import (
	"context"
	"fmt"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

const (
	entryColor = "#f58518"
	nodeColor  = "#4c78a8"
)

// ParseFormat maps a CLI format name to a graphviz format.
func ParseFormat(name string) (graphviz.Format, error) {
	switch name {
	case "png":
		return graphviz.PNG, nil
	case "svg":
		return graphviz.SVG, nil
	}
	return "", fmt.Errorf("unsupported format %q (use png or svg)", name)
}

func Render(ctx context.Context, g *graph.Graph, format graphviz.Format, path string) error {
	gv, err := graphviz.New(ctx)
	if err != nil {
		return err
	}
	defer gv.Close()

	out, err := gv.Graph() // directed by default
	if err != nil {
		return err
	}
	defer out.Close()

	nodes := make(map[string]*cgraph.Node, len(g.Nodes))
	for _, n := range g.Nodes {
		node, err := out.CreateNodeByName(n.ID)
		if err != nil {
			return err
		}
		node.SetShape(cgraph.CircleShape)
		node.SetStyle(cgraph.FilledNodeStyle)
		node.SetFontColor("white")
		if n.Entry {
			node.SetFillColor(entryColor)
		} else {
			node.SetFillColor(nodeColor)
		}
		nodes[n.ID] = node
	}

	for _, e := range g.Edges {
		if _, err := out.CreateEdgeByName(e.From+"->"+e.To, nodes[e.From], nodes[e.To]); err != nil {
			return err
		}
	}

	// Not gv.RenderFilename: in go-graphviz v0.2 it writes through the wasm FS
	// and silently leaves an empty file for absolute paths.
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := gv.Render(ctx, out, format, f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
