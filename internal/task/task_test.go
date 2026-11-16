package task

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestChallengeJSONMarksMasksWithNull(t *testing.T) {
	c := Challenge{ID: "g1-easy-000", Graph: "g1", Level: "easy", Window: NewWindow([]string{"A", "B", "C"}, []int{1})}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"id":"g1-easy-000","graph":"g1","level":"easy","window":["A",null,"C"]}`
	if string(b) != want {
		t.Errorf("json = %s, want %s", b, want)
	}
	var back Challenge
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, c) {
		t.Errorf("round trip = %+v, want %+v", back, c)
	}
	if got := back.Window.MaskPositions(); !reflect.DeepEqual(got, []int{1}) {
		t.Errorf("masks = %v, want [1]", got)
	}
}

func TestNewWindowKeepsItsInput(t *testing.T) {
	tokens := []string{"A", "B"}
	NewWindow(tokens, []int{0})
	if tokens[0] != "A" {
		t.Error("NewWindow modified its input")
	}
}
