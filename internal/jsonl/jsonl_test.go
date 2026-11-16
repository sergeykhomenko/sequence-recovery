package jsonl

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type item struct {
	Name string `json:"name"`
	N    int    `json:"n"`
}

func TestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "items.jsonl")
	want := []item{{"a", 1}, {"b", 2}}
	if err := Write(path, want); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(data); got != "{\"name\":\"a\",\"n\":1}\n{\"name\":\"b\",\"n\":2}\n" {
		t.Errorf("file = %q", got)
	}
	got, err := Read[item](path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
}
