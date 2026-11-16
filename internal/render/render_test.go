package render

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

func TestRenderWritesFile(t *testing.T) {
	g := &graph.Graph{
		Nodes: []graph.Node{{ID: "A", Entry: true}, {ID: "B"}, {ID: "C"}},
		Edges: []graph.Edge{{From: "A", To: "B"}, {From: "B", To: "C"}},
	}
	// t.TempDir is an absolute path, which go-graphviz's RenderFilename mishandles.
	want := map[string][]byte{
		"png": []byte("\x89PNG"),
		"svg": []byte("<svg"),
	}
	for name, marker := range want {
		format, err := ParseFormat(name)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "graph."+name)
		if err := Render(context.Background(), g, format, path); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Contains(data, marker) {
			t.Errorf("%s: output (%d bytes) has no %q marker", name, len(data), marker)
		}
	}
}

func TestParseFormatRejectsUnknown(t *testing.T) {
	if _, err := ParseFormat("gif"); err == nil {
		t.Error("expected error for gif")
	}
}
