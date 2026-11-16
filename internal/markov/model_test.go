package markov

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadModelReadsTrainedCounts(t *testing.T) {
	tr, err := NewTrainer(2)
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		tr.Add([]string{"A", "B"})
	}
	tr.Add([]string{"B", "A"})
	paths, err := tr.Export(filepath.Join(t.TempDir(), "m"))
	if err != nil {
		t.Fatal(err)
	}
	var models []*Model
	for _, p := range paths {
		m, err := LoadModel(p)
		if err != nil {
			t.Fatal(err)
		}
		models = append(models, m)
	}

	// Hand-counted from 3x "A B" and 1x "B A" with gomarkov's ^ / $ padding.
	cases := []struct {
		order   int
		context []string
		want    map[string]int
	}{
		{1, []string{"^"}, map[string]int{"A": 3, "B": 1}},
		{1, []string{"A"}, map[string]int{"B": 3, "$": 1}},
		{2, []string{"^", "^"}, map[string]int{"A": 3, "B": 1}},
		{2, []string{"A", "B"}, map[string]int{"$": 3}},
		{1, []string{"Z"}, nil},
	}
	for _, tc := range cases {
		m := models[tc.order-1]
		if m.Order != tc.order {
			t.Fatalf("model %d has order %d", tc.order, m.Order)
		}
		if got := m.Next(tc.context); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("order %d, Next(%v) = %v, want %v", tc.order, tc.context, got, tc.want)
		}
	}
}
