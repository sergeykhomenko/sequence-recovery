package report

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseMethod(t *testing.T) {
	for method, want := range map[string][2]int{
		MarkovMethod(5, 100): {5, 100},
		"markov-1-k10":       {1, 10},
		"random":             {0, 0},
		"ae":                 {0, 0},
		"markov-3":           {0, 0},
	} {
		if order, budget := ParseMethod(method); order != want[0] || budget != want[1] {
			t.Errorf("ParseMethod(%q) = %d, %d, want %v", method, order, budget, want)
		}
	}
	if got := MarkovMethod(5, 100); got != "markov-5-k100" {
		t.Errorf("MarkovMethod = %q", got)
	}
}

func TestSizeBucket(t *testing.T) {
	for n, want := range map[int]string{16: "16-31", 31: "16-31", 32: "32-63", 100: "64-127", 2048: "2048-4095", 4096: "4096-8191"} {
		if got := SizeBucket(n); got != want {
			t.Errorf("SizeBucket(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestBackoff(t *testing.T) {
	lower, random := Backoff([]int{2, 1, 0, 2}, 2)
	if lower != 0.25 || random != 0.25 {
		t.Errorf("Backoff = %v, %v, want 0.25, 0.25", lower, random)
	}
}

func TestSummarizeGroupsWithTotals(t *testing.T) {
	rows := []Row{
		{Method: "random", Size: 16, Level: "easy", EPA: 0},
		{Method: "random", Size: 20, Level: "easy", EPA: 1},
		{Method: "random", Size: 40, Level: "hard", EPA: 1},
	}
	got := map[string]SummaryRow{}
	for _, s := range Summarize(rows, 10, 1) {
		got[s.Bucket+"/"+s.Level] = s
	}
	want := map[string]struct {
		n   int
		epa float64
	}{
		"16-31/easy": {2, 0.5},
		"16-31/all":  {2, 0.5},
		"32-63/hard": {1, 1},
		"32-63/all":  {1, 1},
		"all/easy":   {2, 0.5},
		"all/hard":   {1, 1},
		"all/all":    {3, 2.0 / 3},
	}
	if len(got) != len(want) {
		t.Fatalf("got groups %v", got)
	}
	for k, w := range want {
		if s := got[k]; s.N != w.n || math.Abs(s.Metrics[0].Mean-w.epa) > 1e-12 {
			t.Errorf("%s: n=%d epa=%v, want n=%d epa=%v", k, s.N, s.Metrics[0].Mean, w.n, w.epa)
		}
	}
}

func TestCompareAgainstReference(t *testing.T) {
	var rows []Row
	for _, id := range []string{"c1", "c2", "c3"} {
		rows = append(rows,
			Row{Method: "ae", Challenge: id, EPA: 1},
			Row{Method: "random", Challenge: id, EPA: 0},
			Row{Method: "markov-1", Challenge: id, EPA: 0},
		)
	}
	got, err := Compare(rows, "ae")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 8 { // 2 other methods x 4 metrics
		t.Fatalf("got %d comparisons", len(got))
	}
	for _, c := range got {
		if c.Metric != "epa" {
			continue
		}
		// Three equal differences: W+ = 6, E = 3, Var = 3.5 - (27-3)/48 = 3, z = √3.
		if c.N != 3 || math.Abs(c.Z-1.7320508076) > 1e-9 || c.PHolm < c.P {
			t.Errorf("%s vs %s: %+v", c.Reference, c.Other, c)
		}
	}
	if none, err := Compare(rows, "missing"); err != nil || len(none) != 0 {
		t.Errorf("missing reference: %v, %v", none, err)
	}
}

func TestWriteRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.csv")
	if err := WriteRows(path, []Row{{Method: "random", Challenge: "c1", EPA: 0.5}}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 || !strings.HasPrefix(lines[0], "method,order,budget,challenge,") || !strings.Contains(lines[1], "0.500000") {
		t.Errorf("csv = %q", data)
	}
}
