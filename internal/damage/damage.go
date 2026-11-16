// Package damage masks positions of a window according to the corruption
// levels of the experiment: easy, medium, hard and extreme.
package damage

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type Level string

const (
	Easy    Level = "easy"
	Medium  Level = "medium"
	Hard    Level = "hard"
	Extreme Level = "extreme"
)

var Levels = []Level{Easy, Medium, Hard, Extreme}

const (
	MinLength = 8 // shortest window the levels are defined for
	MaxGap    = 4 // longest run of consecutive masks
)

// Mask returns the sorted masked positions of a window of the given length.
func Mask(rng *rand.Rand, level Level, length int) ([]int, error) {
	if length < MinLength {
		return nil, fmt.Errorf("window length %d is below %d", length, MinLength)
	}
	var m, gaps int
	switch level {
	case Easy:
		m = 1 + rng.IntN(max(1, min(4, length/20)))
		gaps = 1
	case Medium:
		m = 2 + rng.IntN(max(2, min(8, length/20))-1)
		gaps = 2
	case Hard:
		m = max(1, ratioCount(rng, 0.05, 0.20, length))
		gaps = gapCount(rng, m, length)
	case Extreme:
		m = ratioCount(rng, 0.20, 0.50, length)
		gaps = gapCount(rng, m, length)
	default:
		return nil, fmt.Errorf("unknown level %q", level)
	}
	return place(rng, split(rng, m, gaps), length), nil
}

// Gaps counts the runs of consecutive positions in sorted positions.
func Gaps(positions []int) int {
	n := 0
	for i, p := range positions {
		if i == 0 || p != positions[i-1]+1 {
			n++
		}
	}
	return n
}

// ratioCount draws a share ρ from (lo, hi] and returns round(ρ·length).
func ratioCount(rng *rand.Rand, lo, hi float64, length int) int {
	rho := hi - rng.Float64()*(hi-lo)
	return int(math.Round(rho * float64(length)))
}

// gapCount draws the number of gaps for m masks: at least two when possible,
// enough for runs of at most MaxGap, and no more than fit with separators.
func gapCount(rng *rand.Rand, m, length int) int {
	if m == 1 {
		return 1
	}
	lo := max(2, (m+MaxGap-1)/MaxGap)
	hi := min(m, length-m+1)
	return lo + rng.IntN(hi-lo+1)
}

// split draws g run lengths in [1, MaxGap] that sum to m.
func split(rng *rand.Rand, m, g int) []int {
	runs := make([]int, g)
	for i := range runs {
		runs[i] = 1
	}
	for left := m - g; left > 0; left-- {
		var open []int
		for i, r := range runs {
			if r < MaxGap {
				open = append(open, i)
			}
		}
		runs[open[rng.IntN(len(open))]]++
	}
	return runs
}

// place lays the runs out left to right with random kept stretches between
// them: inner stretches are at least one token long, the outer ones may be empty.
func place(rng *rand.Rand, runs []int, length int) []int {
	m := 0
	for _, r := range runs {
		m += r
	}
	stretches := make([]int, len(runs)+1)
	for i := 1; i < len(runs); i++ {
		stretches[i] = 1
	}
	for left := length - m - (len(runs) - 1); left > 0; left-- {
		stretches[rng.IntN(len(stretches))]++
	}
	pos := make([]int, 0, m)
	at := 0
	for i, r := range runs {
		at += stretches[i]
		for k := range r {
			pos = append(pos, at+k)
		}
		at += r
	}
	return pos
}
