package graph

import (
	"reflect"
	"strings"
	"testing"
)

func TestReadJSON(t *testing.T) {
	// Shape written by cmd/alphabet-graph.
	const in = `{
		"params": {"size": 3, "entries": 1},
		"nodes": [{"id": "A", "entry": true}, {"id": "B", "entry": false}, {"id": "C", "entry": false}],
		"edges": [{"from": "A", "to": "B"}, {"from": "B", "to": "C"}]
	}`

	g, err := ReadJSON(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}

	wantNodes := []Node{{ID: "A", Entry: true}, {ID: "B"}, {ID: "C"}}
	wantEdges := []Edge{{From: "A", To: "B"}, {From: "B", To: "C"}}
	if !reflect.DeepEqual(g.Nodes, wantNodes) {
		t.Errorf("nodes = %+v, want %+v", g.Nodes, wantNodes)
	}
	if !reflect.DeepEqual(g.Edges, wantEdges) {
		t.Errorf("edges = %+v, want %+v", g.Edges, wantEdges)
	}
}
