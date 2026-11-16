package generator

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Strategy chooses prerequisites for the node at index i.
type Strategy interface {
	// Pick returns k distinct indices from [0, i) in ascending order.
	Pick(rng *rand.Rand, i, k int) []int
}

// DecayStrategy weights candidate j by Decay^(i-1-j), so recently added nodes
// are picked more often. Decay = 1 gives a uniform choice.
type DecayStrategy struct {
	Decay float64
}

func (s DecayStrategy) Pick(rng *rand.Rand, i, k int) []int {
	weights := make([]float64, i)
	total := 0.0
	for j := range weights {
		weights[j] = math.Pow(s.Decay, float64(i-1-j))
		total += weights[j]
	}

	taken := make([]bool, i)
	picked := make([]int, 0, k)
	for len(picked) < k {
		r := rng.Float64() * total
		chosen := -1
		for j, w := range weights {
			if taken[j] {
				continue
			}
			chosen = j // falls back to the last free candidate on float rounding/underflow
			if r < w {
				break
			}
			r -= w
		}
		picked = append(picked, chosen)
		taken[chosen] = true
		total -= weights[chosen]
	}

	sort.Ints(picked)
	return picked
}
