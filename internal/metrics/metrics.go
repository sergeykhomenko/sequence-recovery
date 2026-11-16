// Package metrics scores a filled window against the true one (spec section 5).
package metrics

import (
	"errors"
	"fmt"
	"slices"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

type Result struct {
	EPA, OAR, UAR, PVR float64
	Valid              bool
}

// Score checks that fill obeys the recovery rules and computes the metrics.
func Score(truth []string, w task.Window, fill []string, cl *graph.Closure) (Result, error) {
	masked := w.MaskPositions()
	switch {
	case len(truth) != len(w):
		return Result{}, fmt.Errorf("window has %d tokens, truth %d", len(w), len(truth))
	case len(masked) == 0:
		return Result{}, errors.New("window has no masks")
	case len(fill) != len(masked):
		return Result{}, fmt.Errorf("got %d tokens for %d masks", len(fill), len(masked))
	}

	rec := slices.Clone(w)
	for k, i := range masked {
		rec[i] = fill[k]
	}
	seen := make(map[string]bool, len(rec))
	for _, t := range rec {
		if !cl.Has(t) {
			return Result{}, fmt.Errorf("symbol %q is not in the alphabet", t)
		}
		if seen[t] {
			return Result{}, fmt.Errorf("symbol %q appears twice", t)
		}
		seen[t] = true
	}

	want := make([]string, len(masked))
	exact, violations := 0, 0
	for k, i := range masked {
		want[k] = truth[i]
		if fill[k] == truth[i] {
			exact++
		}
		if violates(rec, i, cl) {
			violations++
		}
	}
	m := float64(len(masked))
	return Result{
		EPA:   float64(exact) / m,
		OAR:   float64(lcs(want, fill)) / m,
		UAR:   float64(common(want, fill)) / m,
		PVR:   float64(violations) / m,
		Valid: violations == 0,
	}, nil
}

// violates reports whether rec[i] has an ancestor to its right or a descendant to its left.
func violates(rec []string, i int, cl *graph.Closure) bool {
	x := rec[i]
	for j, y := range rec {
		if (j > i && cl.Precedes(y, x)) || (j < i && cl.Precedes(x, y)) {
			return true
		}
	}
	return false
}

func lcs(a, b []string) int {
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for i := range a {
		for j := range b {
			if a[i] == b[j] {
				cur[j+1] = prev[j] + 1
			} else {
				cur[j+1] = max(prev[j+1], cur[j])
			}
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

// common counts the tokens shared by a and b as multisets.
func common(a, b []string) int {
	count := make(map[string]int, len(a))
	for _, t := range a {
		count[t]++
	}
	n := 0
	for _, t := range b {
		if count[t] > 0 {
			count[t]--
			n++
		}
	}
	return n
}
