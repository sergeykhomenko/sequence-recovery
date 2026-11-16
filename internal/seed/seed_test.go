package seed

import "testing"

func TestDeriveIsStable(t *testing.T) {
	// Fixed reference value protects dataset reproducibility across releases.
	const want uint64 = 10359997354575632654
	if got := Derive(1, "graph", "g0001"); got != want {
		t.Errorf("derived seed = %d, want %d", got, want)
	}
}

func TestDeriveSeparatesStreams(t *testing.T) {
	seeds := map[string]uint64{
		"base":        Derive(1, "graph", "g0001"),
		"other base":  Derive(2, "graph", "g0001"),
		"other id":    Derive(1, "graph", "g0002"),
		"other stage": Derive(1, "orders", "g0001"),
		"joined":      Derive(1, "graphg0001"),
		"split":       Derive(1, "gra", "phg0001"),
	}
	seen := map[uint64]string{}
	for name, s := range seeds {
		if other, dup := seen[s]; dup {
			t.Errorf("%s and %s share seed %d", name, other, s)
		}
		seen[s] = name
	}
}

func TestRNGIsReproducible(t *testing.T) {
	a, b := RNG(7, "x"), RNG(7, "x")
	for range 10 {
		if a.Uint64() != b.Uint64() {
			t.Fatal("same stream diverged")
		}
	}
}
