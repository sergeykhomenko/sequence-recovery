// Package alphabet names graph nodes with letters: A..Z, AA..AZ, BA, ...
package alphabet

// Name returns the 1-based i-th letter name (bijective base-26, like spreadsheet columns).
func Name(i int) string {
	var b []byte
	for i > 0 {
		i--
		b = append(b, byte('A'+i%26))
		i /= 26
	}
	for l, r := 0, len(b)-1; l < r; l, r = l+1, r-1 {
		b[l], b[r] = b[r], b[l]
	}
	return string(b)
}

// Letters returns the first n letter names.
func Letters(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = Name(i + 1)
	}
	return out
}
