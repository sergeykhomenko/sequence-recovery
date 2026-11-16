package store

import (
	"reflect"
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/experiment"
	"github.com/sergeykhomenko/sequence-recovery/internal/generator"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

func newStore(t *testing.T) Store {
	st := Store{Dir: t.TempDir()}
	if err := st.Init(); err != nil {
		t.Fatal(err)
	}
	return st
}

func TestGraphRoundTrip(t *testing.T) {
	st := newStore(t)
	g := &graph.Graph{
		Nodes: []graph.Node{{ID: "K2Q", Entry: true}, {ID: "ZZ1"}},
		Edges: []graph.Edge{{From: "K2Q", To: "ZZ1"}},
	}
	p := GraphParams{ID: "g0001", Split: Test, Config: generator.Config{Size: 2, Entries: 1, MinPrereq: 1, MaxPrereq: 1, Decay: 0.8, Seed: 9}}
	if err := st.WriteGraph(g, p); err != nil {
		t.Fatal(err)
	}
	got, params, err := st.ReadGraph("g0001")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Nodes, g.Nodes) || !reflect.DeepEqual(got.Edges, g.Edges) {
		t.Errorf("graph = %+v, want %+v", got, g)
	}
	if params != p {
		t.Errorf("params = %+v, want %+v", params, p)
	}
}

func TestGraphIDsFiltersBySplit(t *testing.T) {
	st := newStore(t)
	for _, p := range []GraphParams{{ID: "g0003", Split: Test}, {ID: "g0001", Split: Train}, {ID: "g0002", Split: Test}} {
		if err := st.WriteGraph(&graph.Graph{}, p); err != nil {
			t.Fatal(err)
		}
	}
	ids, err := st.GraphIDs(Test)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"g0002", "g0003"}; !reflect.DeepEqual(ids, want) {
		t.Errorf("ids = %v, want %v", ids, want)
	}
}

func TestOrdersRoundTrip(t *testing.T) {
	st := newStore(t)
	orders := [][]string{{"K2Q", "ZZ1"}, {"ZZ1", "K2Q"}}
	if err := st.WriteOrders("g0001", orders); err != nil {
		t.Fatal(err)
	}
	got, err := st.ReadOrders("g0001")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, orders) {
		t.Errorf("orders = %v, want %v", got, orders)
	}
}

func TestConfigRoundTrip(t *testing.T) {
	st := newStore(t)
	c, err := experiment.Load("../../experiments/smoke.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := st.WriteConfig(c); err != nil {
		t.Fatal(err)
	}
	got, err := st.ReadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, c) {
		t.Errorf("config = %+v, want %+v", got, c)
	}
}
