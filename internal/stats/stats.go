// Package stats: bootstrap confidence intervals, the paired Wilcoxon
// signed-rank test and Holm's correction for multiple comparisons.
package stats

import (
	"cmp"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

// Bootstrap returns the mean and the 95% percentile interval of the mean.
func Bootstrap(values []float64, iters int, rng *rand.Rand) (mean, lo, hi float64) {
	if len(values) == 0 {
		return math.NaN(), math.NaN(), math.NaN()
	}
	mean = average(values)
	if iters < 1 {
		return mean, mean, mean
	}
	means := make([]float64, iters)
	for i := range means {
		s := 0.0
		for range values {
			s += values[rng.IntN(len(values))]
		}
		means[i] = s / float64(len(values))
	}
	slices.Sort(means)
	return mean, quantile(means, 0.025), quantile(means, 0.975)
}

type Wilcoxon struct {
	N          int     // pairs with a non-zero difference
	MedianDiff float64 // median of x-y over all pairs
	Z, P, R    float64 // normal statistic, two-sided p, effect size z/√N
}

// SignedRank is the two-sided paired Wilcoxon signed-rank test with the normal
// approximation and tie correction; zero differences are dropped.
func SignedRank(x, y []float64) (Wilcoxon, error) {
	if len(x) != len(y) {
		return Wilcoxon{}, errors.New("samples differ in length")
	}
	diffs := make([]float64, len(x))
	var nz []float64
	for i := range x {
		diffs[i] = x[i] - y[i]
		if diffs[i] != 0 {
			nz = append(nz, diffs[i])
		}
	}
	res := Wilcoxon{N: len(nz), MedianDiff: median(diffs), P: 1}
	n := len(nz)
	if n == 0 {
		return res, nil
	}
	slices.SortFunc(nz, func(a, b float64) int { return cmp.Compare(math.Abs(a), math.Abs(b)) })
	var wPlus, ties float64
	for i := 0; i < n; {
		j := i
		for j < n && math.Abs(nz[j]) == math.Abs(nz[i]) {
			j++
		}
		rank := float64(i+j+1) / 2 // average of ranks i+1..j
		for k := i; k < j; k++ {
			if nz[k] > 0 {
				wPlus += rank
			}
		}
		t := float64(j - i)
		ties += t*t*t - t
		i = j
	}
	nf := float64(n)
	variance := nf*(nf+1)*(2*nf+1)/24 - ties/48
	if variance <= 0 {
		return res, nil
	}
	res.Z = (wPlus - nf*(nf+1)/4) / math.Sqrt(variance)
	res.P = math.Erfc(math.Abs(res.Z) / math.Sqrt2)
	res.R = res.Z / math.Sqrt(nf)
	return res, nil
}

// Holm adjusts p-values for multiple comparisons (step-down, kept monotone).
func Holm(p []float64) []float64 {
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return p[idx[a]] < p[idx[b]] })
	out := make([]float64, len(p))
	prev := 0.0
	for rank, i := range idx {
		prev = math.Max(prev, math.Min(1, float64(len(p)-rank)*p[i]))
		out[i] = prev
	}
	return out
}

func average(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	s := slices.Clone(v)
	slices.Sort(s)
	if n := len(s); n%2 == 1 {
		return s[n/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// quantile interpolates linearly between the closest ranks of a sorted sample.
func quantile(sorted []float64, q float64) float64 {
	pos := q * float64(len(sorted)-1)
	i := int(pos)
	if i+1 >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	return sorted[i] + (pos-float64(i))*(sorted[i+1]-sorted[i])
}
