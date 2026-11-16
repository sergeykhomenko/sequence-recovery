package alphabet

import "testing"

func TestName(t *testing.T) {
	cases := map[int]string{
		1:   "A",
		26:  "Z",
		27:  "AA",
		52:  "AZ",
		53:  "BA",
		702: "ZZ",
		703: "AAA",
	}
	for i, want := range cases {
		if got := Name(i); got != want {
			t.Errorf("Name(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestLetters(t *testing.T) {
	got := Letters(3)
	want := []string{"A", "B", "C"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Letters(3) = %v, want %v", got, want)
		}
	}
}
