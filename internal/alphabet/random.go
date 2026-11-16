package alphabet

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
)

// Random returns n distinct random symbols of the given length over an ASCII charset.
func Random(rng *rand.Rand, n int, charset string, length int) ([]string, error) {
	if length < 1 || len(charset) < 2 {
		return nil, errors.New("need a charset of at least 2 characters and length >= 1")
	}
	if math.Pow(float64(len(charset)), float64(length)) < float64(n) {
		return nil, fmt.Errorf("%d characters of length %d cannot name %d symbols", len(charset), length, n)
	}
	seen := make(map[string]bool, n)
	out := make([]string, 0, n)
	buf := make([]byte, length)
	for len(out) < n {
		for i := range buf {
			buf[i] = charset[rng.IntN(len(charset))]
		}
		if s := string(buf); !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out, nil
}
