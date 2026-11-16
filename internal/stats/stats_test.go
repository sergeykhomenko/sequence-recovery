package stats

import (
	"math"
	"math/rand/v2"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSignedRankWithoutTies(t *testing.T) {
	// |d| ranks 1..8; positives hold ranks 2,3,4,6,7,8, so W+ = 30, E = 18, Var = 51.
	x := []float64{1.5, -0.5, 2.5, 3.5, -4.5, 5.5, 6.5, 7.5}
	w, err := SignedRank(x, make([]float64, len(x)))
	if err != nil {
		t.Fatal(err)
	}
	if w.N != 8 || !near(w.Z, 1.6803361008) || !near(w.P, 0.0928919409) || !near(w.R, 0.5940885258) || w.MedianDiff != 3 {
		t.Errorf("got %+v", w)
	}
}

func TestSignedRankWithTiesAndZeros(t *testing.T) {
	// The zero is dropped: n = 4, W+ = 1.5+1.5+4 = 7, Var = 7.5 - (2³-2)/48 = 7.375.
	w, err := SignedRank([]float64{1, 1, -2, 3, 0}, make([]float64, 5))
	if err != nil {
		t.Fatal(err)
	}
	if w.N != 4 || !near(w.Z, 0.7364596943) || !near(w.P, 0.4614509878) || w.MedianDiff != 1 {
		t.Errorf("got %+v", w)
	}
}

func TestSignedRankEdgeCases(t *testing.T) {
	w, err := SignedRank([]float64{1, 2}, []float64{1, 2})
	if err != nil || w.N != 0 || w.P != 1 {
		t.Errorf("identical samples: %+v, %v", w, err)
	}
	if _, err := SignedRank([]float64{1}, nil); err == nil {
		t.Error("samples of different length: no error")
	}
}

func TestHolm(t *testing.T) {
	got := Holm([]float64{0.01, 0.04, 0.03})
	want := []float64{0.03, 0.06, 0.06}
	for i := range want {
		if !near(got[i], want[i]) {
			t.Fatalf("Holm = %v, want %v", got, want)
		}
	}
}

func TestBootstrap(t *testing.T) {
	mean, lo, hi := Bootstrap([]float64{0.5, 0.5, 0.5}, 100, rand.New(rand.NewPCG(1, 1)))
	if mean != 0.5 || lo != 0.5 || hi != 0.5 {
		t.Errorf("constant sample: %v [%v, %v]", mean, lo, hi)
	}

	values := make([]float64, 100)
	for i := range 50 {
		values[i] = 1
	}
	// Normal approximation of the 95% interval: 0.5 ± 1.96·0.5/√100 = [0.402, 0.598].
	mean, lo, hi = Bootstrap(values, 2000, rand.New(rand.NewPCG(1, 1)))
	if mean != 0.5 || math.Abs(lo-0.402) > 0.03 || math.Abs(hi-0.598) > 0.03 {
		t.Errorf("0/1 sample: %v [%v, %v]", mean, lo, hi)
	}
	_, lo2, hi2 := Bootstrap(values, 2000, rand.New(rand.NewPCG(1, 1)))
	if lo2 != lo || hi2 != hi {
		t.Error("same seed gave a different interval")
	}
}
