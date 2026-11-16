package graph

import "testing"

func TestClosure(t *testing.T) {
	g := &Graph{
		Nodes: []Node{{ID: "A"}, {ID: "B"}, {ID: "C"}, {ID: "D"}},
		Edges: []Edge{{From: "A", To: "B"}, {From: "B", To: "C"}},
	}
	c, err := NewClosure(g)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		a, b string
		want bool
	}{
		{"A", "B", true},
		{"A", "C", true}, // transitive
		{"C", "A", false},
		{"A", "A", false},
		{"D", "C", false},
		{"A", "Z", false}, // unknown node
	}
	for _, tc := range cases {
		if got := c.Precedes(tc.a, tc.b); got != tc.want {
			t.Errorf("Precedes(%s, %s) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
	if c.Has("Z") || !c.Has("D") {
		t.Error("Has is wrong")
	}
}

func TestClosureRejectsBadGraphs(t *testing.T) {
	for name, g := range map[string]*Graph{
		"cycle":        {Nodes: []Node{{ID: "A"}, {ID: "B"}}, Edges: []Edge{{From: "A", To: "B"}, {From: "B", To: "A"}}},
		"unknown node": {Nodes: []Node{{ID: "A"}}, Edges: []Edge{{From: "A", To: "X"}}},
	} {
		if _, err := NewClosure(g); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
