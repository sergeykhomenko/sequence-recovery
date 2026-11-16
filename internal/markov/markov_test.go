package markov

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mb-14/gomarkov"
)

func TestExportedChainsHoldTrainedTransitions(t *testing.T) {
	tr, err := NewTrainer(2)
	if err != nil {
		t.Fatal(err)
	}
	for range 3 {
		tr.Add([]string{"A", "B"})
	}
	tr.Add([]string{"B", "A"})

	prefix := filepath.Join(t.TempDir(), "markov")
	paths, err := tr.Export(prefix)
	if err != nil {
		t.Fatal(err)
	}
	wantPaths := []string{prefix + "_1.json", prefix + "_2.json"}
	if !reflect.DeepEqual(paths, wantPaths) {
		t.Fatalf("paths = %v, want %v", paths, wantPaths)
	}

	// Hand-counted from 3x "A B" and 1x "B A", with gomarkov's ^ / $ padding.
	cases := []struct {
		order   int
		current gomarkov.NGram
		next    string
		want    float64
	}{
		{1, gomarkov.NGram{"^"}, "A", 0.75},
		{1, gomarkov.NGram{"A"}, "B", 0.75},
		{1, gomarkov.NGram{"B"}, "$", 0.75},
		{2, gomarkov.NGram{"^", "^"}, "A", 0.75},
		{2, gomarkov.NGram{"^", "A"}, "B", 1},
		{2, gomarkov.NGram{"^", "B"}, "A", 1},
	}
	chains := map[int]*gomarkov.Chain{}
	for i, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var c gomarkov.Chain
		if err := c.UnmarshalJSON(data); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if c.Order != i+1 {
			t.Fatalf("%s holds order %d, want %d", path, c.Order, i+1)
		}
		chains[c.Order] = &c
	}
	for _, tc := range cases {
		got, err := chains[tc.order].TransitionProbability(tc.next, tc.current)
		if err != nil {
			t.Fatal(err)
		}
		if got != tc.want {
			t.Errorf("order %d: P(%s | %v) = %v, want %v", tc.order, tc.next, tc.current, got, tc.want)
		}
	}
}

func TestNewTrainerRejectsNonPositiveOrder(t *testing.T) {
	if _, err := NewTrainer(0); err == nil {
		t.Error("NewTrainer(0) returned no error")
	}
}
