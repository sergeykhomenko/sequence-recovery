package alphabet

import (
	"math/rand/v2"
	"strings"
	"testing"
)

func TestRandom(t *testing.T) {
	syms, err := Random(rand.New(rand.NewPCG(1, 1)), 500, "AB12", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(syms) != 500 {
		t.Fatalf("got %d symbols, want 500", len(syms))
	}
	seen := map[string]bool{}
	for _, s := range syms {
		if len(s) != 5 || strings.Trim(s, "AB12") != "" {
			t.Errorf("symbol %q is not 5 characters of AB12", s)
		}
		if seen[s] {
			t.Errorf("symbol %q repeats", s)
		}
		seen[s] = true
	}
}

func TestRandomRejectsTooSmallSpace(t *testing.T) {
	// 2^4 = 16 symbols exist, 17 are asked for.
	if _, err := Random(rand.New(rand.NewPCG(1, 1)), 17, "AB", 4); err == nil {
		t.Error("expected an error")
	}
}
