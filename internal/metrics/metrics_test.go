package metrics

import (
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

// closure of the chain A -> B -> C -> D plus an independent E.
func closure(t *testing.T) *graph.Closure {
	t.Helper()
	g := &graph.Graph{
		Nodes: []graph.Node{{ID: "A"}, {ID: "B"}, {ID: "C"}, {ID: "D"}, {ID: "E"}},
		Edges: []graph.Edge{{From: "A", To: "B"}, {From: "B", To: "C"}, {From: "C", To: "D"}},
	}
	c, err := graph.NewClosure(g)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestScore(t *testing.T) {
	cl := closure(t)
	truth := []string{"A", "B", "C", "D"}
	window := task.Window{"A", "", "", "D"}
	cases := []struct {
		name string
		fill []string
		want Result
	}{
		{"exact", []string{"B", "C"}, Result{EPA: 1, OAR: 1, UAR: 1, PVR: 0, Valid: true}},
		// Right symbols, wrong order: both inserts break B -> C.
		{"swapped", []string{"C", "B"}, Result{EPA: 0, OAR: 0.5, UAR: 1, PVR: 1, Valid: false}},
		// E has no dependencies, B stays after A and before D: wrong but valid.
		{"valid but wrong", []string{"E", "B"}, Result{EPA: 0, OAR: 0.5, UAR: 0.5, PVR: 0, Valid: true}},
	}
	for _, tc := range cases {
		got, err := Score(truth, window, tc.fill, cl)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.name, got, tc.want)
		}
	}
}

func TestScoreRejectsRuleBreaks(t *testing.T) {
	cl := closure(t)
	truth := []string{"A", "B", "C", "D"}
	window := task.Window{"A", "", "", "D"}
	for name, fill := range map[string][]string{
		"too few tokens": {"B"},
		"unknown symbol": {"B", "Z"},
		"repeats kept A": {"A", "C"},
		"repeats itself": {"B", "B"},
	} {
		if _, err := Score(truth, window, fill, cl); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := Score(truth, task.Window{"A", "B", "C", "D"}, nil, cl); err == nil {
		t.Error("window without masks: no error")
	}
}
