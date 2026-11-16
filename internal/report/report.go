// Package report turns scored challenges into the experiment's CSV files.
package report

import (
	"cmp"
	"encoding/csv"
	"fmt"
	"math"
	"math/bits"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/sergeykhomenko/sequence-recovery/internal/damage"
	"github.com/sergeykhomenko/sequence-recovery/internal/seed"
	"github.com/sergeykhomenko/sequence-recovery/internal/stats"
)

// Row is one challenge scored for one method.
type Row struct {
	Method    string
	Order     int // Markov order, 0 for other methods
	Budget    int // Markov training trajectories, 0 for other methods
	Challenge string
	Graph     string
	Size      int // alphabet size N
	Length    int // window length L
	Level     string
	Ratio     float64 // masks / L
	Gaps      int

	EPA, OAR, UAR, PVR, VWR float64
	Backoff, Fallback       float64 // Markov only
}

var metricNames = [5]string{"epa", "oar", "uar", "pvr", "vwr"}

func (r Row) metrics() [5]float64 { return [5]float64{r.EPA, r.OAR, r.UAR, r.PVR, r.VWR} }

// MarkovMethod names the predictions of a Markov order trained on budget trajectories.
func MarkovMethod(order, budget int) string {
	return fmt.Sprintf("markov-%d-k%d", order, budget)
}

// ParseMethod extracts order and budget from a MarkovMethod name; other methods give 0, 0.
func ParseMethod(method string) (order, budget int) {
	if _, err := fmt.Sscanf(method, "markov-%d-k%d", &order, &budget); err != nil {
		return 0, 0
	}
	return order, budget
}

// SizeBucket names the power-of-two range of an alphabet size, e.g. 16-31.
func SizeBucket(n int) string {
	lo := 1 << (bits.Len(uint(n)) - 1)
	return fmt.Sprintf("%d-%d", lo, 2*lo-1)
}

// Backoff returns the shares of masks filled from a lower order than asked
// (1 <= used < order) and at random (used == 0).
func Backoff(used []int, order int) (lower, random float64) {
	for _, u := range used {
		switch {
		case u == 0:
			random++
		case u < order:
			lower++
		}
	}
	n := float64(len(used))
	return lower / n, random / n
}

type Cell struct {
	Mean, Lo, Hi float64
}

type SummaryRow struct {
	Method  string
	Order   int
	Budget  int
	Bucket  string // size bucket or "all"
	Level   string // damage level or "all"
	N       int
	Metrics [5]Cell // EPA, OAR, UAR, PVR, VWR
}

// Summarize averages every metric with a 95% bootstrap interval per method,
// size bucket and level, including "all" rows over buckets and levels.
func Summarize(rows []Row, iters int, seedValue uint64) []SummaryRow {
	type key struct{ method, bucket, level string }
	groups := map[key][]Row{}
	for _, r := range rows {
		for _, b := range []string{SizeBucket(r.Size), "all"} {
			for _, l := range []string{r.Level, "all"} {
				k := key{r.Method, b, l}
				groups[k] = append(groups[k], r)
			}
		}
	}
	keys := make([]key, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b key) int {
		return cmp.Or(
			cmp.Compare(a.method, b.method),
			cmp.Compare(bucketLow(a.bucket), bucketLow(b.bucket)),
			cmp.Compare(levelRank(a.level), levelRank(b.level)),
		)
	})

	out := make([]SummaryRow, 0, len(keys))
	for _, k := range keys {
		g := groups[k]
		order, budget := ParseMethod(k.method)
		s := SummaryRow{Method: k.method, Order: order, Budget: budget, Bucket: k.bucket, Level: k.level, N: len(g)}
		for m, name := range metricNames {
			vals := make([]float64, len(g))
			for i, r := range g {
				vals[i] = r.metrics()[m]
			}
			rng := seed.RNG(seedValue, "bootstrap", k.method, k.bucket, k.level, name)
			mean, lo, hi := stats.Bootstrap(vals, iters, rng)
			s.Metrics[m] = Cell{mean, lo, hi}
		}
		out = append(out, s)
	}
	return out
}

func bucketLow(b string) int {
	lo, _, _ := strings.Cut(b, "-")
	if v, err := strconv.Atoi(lo); err == nil {
		return v
	}
	return math.MaxInt // "all" goes last
}

func levelRank(l string) int {
	for i, lv := range damage.Levels {
		if string(lv) == l {
			return i
		}
	}
	return len(damage.Levels) // "all" goes last
}

type Comparison struct {
	Reference, Other, Metric string
	stats.Wilcoxon
	PHolm float64
}

// Compare runs the paired signed-rank test of reference against every other
// method on the challenges both scored, for EPA, OAR, UAR and PVR, with Holm's
// correction across methods within a metric. No reference rows, no comparisons.
func Compare(rows []Row, reference string) ([]Comparison, error) {
	byMethod := map[string]map[string]Row{}
	for _, r := range rows {
		if byMethod[r.Method] == nil {
			byMethod[r.Method] = map[string]Row{}
		}
		byMethod[r.Method][r.Challenge] = r
	}
	ref, ok := byMethod[reference]
	if !ok {
		return nil, nil
	}
	var others []string
	for m := range byMethod {
		if m != reference {
			others = append(others, m)
		}
	}
	slices.Sort(others)
	ids := make([]string, 0, len(ref))
	for id := range ref {
		ids = append(ids, id)
	}
	slices.Sort(ids)

	var out []Comparison
	for m, name := range metricNames[:4] {
		start := len(out)
		for _, o := range others {
			var x, y []float64
			for _, id := range ids {
				if r, ok := byMethod[o][id]; ok {
					x = append(x, ref[id].metrics()[m])
					y = append(y, r.metrics()[m])
				}
			}
			w, err := stats.SignedRank(x, y)
			if err != nil {
				return nil, err
			}
			out = append(out, Comparison{Reference: reference, Other: o, Metric: name, Wilcoxon: w})
		}
		ps := make([]float64, len(out)-start)
		for i := range ps {
			ps[i] = out[start+i].P
		}
		for i, p := range stats.Holm(ps) {
			out[start+i].PHolm = p
		}
	}
	return out, nil
}

func f(v float64) string { return strconv.FormatFloat(v, 'f', 6, 64) }

func writeCSV(path string, header []string, records [][]string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(file)
	w.Write(header)
	w.WriteAll(records) // flushes
	if err := w.Error(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func WriteRows(path string, rows []Row) error {
	header := []string{"method", "order", "budget", "challenge", "graph", "size", "length", "level", "ratio", "gaps",
		"epa", "oar", "uar", "pvr", "vwr", "backoff", "fallback"}
	records := make([][]string, len(rows))
	for i, r := range rows {
		records[i] = []string{r.Method, strconv.Itoa(r.Order), strconv.Itoa(r.Budget), r.Challenge, r.Graph,
			strconv.Itoa(r.Size), strconv.Itoa(r.Length), r.Level, f(r.Ratio), strconv.Itoa(r.Gaps),
			f(r.EPA), f(r.OAR), f(r.UAR), f(r.PVR), f(r.VWR), f(r.Backoff), f(r.Fallback)}
	}
	return writeCSV(path, header, records)
}

func WriteSummary(path string, rows []SummaryRow) error {
	header := []string{"method", "order", "budget", "bucket", "level", "n"}
	for _, name := range metricNames {
		header = append(header, name+"_mean", name+"_lo", name+"_hi")
	}
	records := make([][]string, len(rows))
	for i, s := range rows {
		rec := []string{s.Method, strconv.Itoa(s.Order), strconv.Itoa(s.Budget), s.Bucket, s.Level, strconv.Itoa(s.N)}
		for _, c := range s.Metrics {
			rec = append(rec, f(c.Mean), f(c.Lo), f(c.Hi))
		}
		records[i] = rec
	}
	return writeCSV(path, header, records)
}

func WriteComparisons(path string, rows []Comparison) error {
	header := []string{"reference", "other", "metric", "n", "median_diff", "z", "p", "p_holm", "r"}
	records := make([][]string, len(rows))
	for i, c := range rows {
		records[i] = []string{c.Reference, c.Other, c.Metric, strconv.Itoa(c.N),
			f(c.MedianDiff), f(c.Z), f(c.P), f(c.PHolm), f(c.R)}
	}
	return writeCSV(path, header, records)
}
