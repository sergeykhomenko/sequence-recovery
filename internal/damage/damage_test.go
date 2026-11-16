package damage

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// runLengths returns the lengths of consecutive runs in sorted positions.
func runLengths(pos []int) []int {
	var out []int
	for i, p := range pos {
		if i == 0 || p != pos[i-1]+1 {
			out = append(out, 0)
		}
		out[len(out)-1]++
	}
	return out
}

func TestGaps(t *testing.T) {
	if got := Gaps([]int{0, 1, 2, 5, 7, 8}); got != 3 {
		t.Errorf("Gaps = %d, want 3", got)
	}
	if got := Gaps(nil); got != 0 {
		t.Errorf("Gaps(nil) = %d, want 0", got)
	}
}

func TestMaskOnShortestWindow(t *testing.T) {
	// L = 8: ⌊0.05·8⌋ = 0, so easy masks exactly one position and medium exactly two, apart.
	for s := range uint64(50) {
		rng := rand.New(rand.NewPCG(s, s))
		easy, err := Mask(rng, Easy, 8)
		if err != nil {
			t.Fatal(err)
		}
		if len(easy) != 1 {
			t.Errorf("easy masked %v", easy)
		}
		medium, err := Mask(rng, Medium, 8)
		if err != nil {
			t.Fatal(err)
		}
		if len(medium) != 2 || Gaps(medium) != 2 {
			t.Errorf("medium masked %v", medium)
		}
	}
}

func TestMaskFollowsLevelRules(t *testing.T) {
	round := func(x float64) int { return int(math.Round(x)) }
	for _, length := range []int{8, 9, 20, 41, 100, 256} {
		// Mask-count bounds from the spec, section 3.5.
		bounds := map[Level][2]int{
			Easy:    {1, max(1, min(4, length/20))},
			Medium:  {2, max(2, min(8, length/20))},
			Hard:    {max(1, round(0.05*float64(length))), max(1, round(0.20*float64(length)))},
			Extreme: {max(2, round(0.20*float64(length))), round(0.50 * float64(length))},
		}
		for s := range uint64(300) {
			rng := rand.New(rand.NewPCG(s, uint64(length)))
			for _, level := range Levels {
				pos, err := Mask(rng, level, length)
				if err != nil {
					t.Fatalf("%s L=%d: %v", level, length, err)
				}
				if !slices.IsSorted(pos) || len(slices.Compact(slices.Clone(pos))) != len(pos) {
					t.Fatalf("%s L=%d: positions %v are not sorted and unique", level, length, pos)
				}
				if pos[0] < 0 || pos[len(pos)-1] >= length {
					t.Fatalf("%s L=%d: positions %v leave the window", level, length, pos)
				}
				if m := len(pos); m < bounds[level][0] || m > bounds[level][1] {
					t.Errorf("%s L=%d: %d masks, want [%d, %d]", level, length, m, bounds[level][0], bounds[level][1])
				}
				runs := runLengths(pos)
				for _, r := range runs {
					if r > MaxGap {
						t.Errorf("%s L=%d: run of %d masks in %v", level, length, r, pos)
					}
				}
				switch {
				case level == Easy && len(runs) != 1,
					level == Medium && len(runs) != 2,
					(level == Hard || level == Extreme) && len(pos) >= 2 && len(runs) < 2:
					t.Errorf("%s L=%d: %d gaps in %v", level, length, len(runs), pos)
				}
			}
		}
	}
}

func TestMaskReachesWindowEdges(t *testing.T) {
	first, last := false, false
	for s := range uint64(200) {
		pos, err := Mask(rand.New(rand.NewPCG(s, 1)), Extreme, 20)
		if err != nil {
			t.Fatal(err)
		}
		first = first || pos[0] == 0
		last = last || pos[len(pos)-1] == 19
	}
	if !first || !last {
		t.Errorf("window edges never masked: first=%v last=%v", first, last)
	}
}

func TestMaskRejectsBadInput(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	if _, err := Mask(rng, Easy, 7); err == nil {
		t.Error("accepted a window of 7")
	}
	if _, err := Mask(rng, Level("brutal"), 20); err == nil {
		t.Error("accepted an unknown level")
	}
}
