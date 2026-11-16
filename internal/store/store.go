// Package store is the on-disk layout of one experiment directory.
package store

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sergeykhomenko/sequence-recovery/internal/experiment"
	"github.com/sergeykhomenko/sequence-recovery/internal/generator"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
)

const (
	Train = "train"
	Val   = "val"
	Test  = "test"
)

// GraphParams is written as the "params" of every graph file.
type GraphParams struct {
	ID    string `json:"id"`
	Split string `json:"split"`
	generator.Config
}

type Store struct {
	Dir string
}

func (s Store) path(parts ...string) string {
	return filepath.Join(append([]string{s.Dir}, parts...)...)
}

func (s Store) ConfigPath() string          { return s.path("config.json") }
func (s Store) GraphPath(id string) string  { return s.path("graphs", id+".json") }
func (s Store) OrdersPath(id string) string { return s.path("orders", id+".txt") }
func (s Store) MarkovPrefix(budget int, id string) string {
	return s.path("markov", fmt.Sprintf("k%d", budget), id)
}
func (s Store) ChallengesPath() string              { return s.path("challenges.jsonl") }
func (s Store) AnswersPath() string                 { return s.path("answers.jsonl") }
func (s Store) AETrainPath() string                 { return s.path("ae-train.jsonl") }
func (s Store) AEValPath() string                   { return s.path("ae-val.jsonl") }
func (s Store) PredictionsDir() string              { return s.path("predictions") }
func (s Store) PredictionPath(method string) string { return s.path("predictions", method+".jsonl") }
func (s Store) MetricsPath() string                 { return s.path("metrics.csv") }
func (s Store) SummaryPath() string                 { return s.path("summary.csv") }
func (s Store) ComparisonsPath() string             { return s.path("comparisons.csv") }

// Init creates the directory tree of the experiment.
func (s Store) Init() error {
	for _, d := range []string{"graphs", "orders", "markov", "predictions"} {
		if err := os.MkdirAll(s.path(d), 0o755); err != nil {
			return err
		}
	}
	return nil
}

func (s Store) WriteConfig(c experiment.Config) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.ConfigPath(), append(data, '\n'), 0o644)
}

func (s Store) ReadConfig() (experiment.Config, error) {
	return experiment.Load(s.ConfigPath())
}

func (s Store) WriteGraph(g *graph.Graph, p GraphParams) error {
	out := *g
	out.Params = p
	f, err := os.Create(s.GraphPath(p.ID))
	if err != nil {
		return err
	}
	if err := out.WriteJSON(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

type graphFile struct {
	Params GraphParams  `json:"params"`
	Nodes  []graph.Node `json:"nodes"`
	Edges  []graph.Edge `json:"edges"`
}

func (s Store) ReadGraph(id string) (*graph.Graph, GraphParams, error) {
	data, err := os.ReadFile(s.GraphPath(id))
	if err != nil {
		return nil, GraphParams{}, err
	}
	var f graphFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, GraphParams{}, fmt.Errorf("%s: %w", s.GraphPath(id), err)
	}
	return &graph.Graph{Params: f.Params, Nodes: f.Nodes, Edges: f.Edges}, f.Params, nil
}

// GraphIDs lists the graphs of one split, sorted by ID.
func (s Store) GraphIDs(split string) ([]string, error) {
	paths, err := filepath.Glob(s.path("graphs", "*.json"))
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, p := range paths {
		id := strings.TrimSuffix(filepath.Base(p), ".json")
		_, params, err := s.ReadGraph(id)
		if err != nil {
			return nil, err
		}
		if params.Split == split {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids, nil
}

// WriteOrders writes one trajectory per line, tokens separated by spaces.
func (s Store) WriteOrders(id string, orders [][]string) error {
	f, err := os.Create(s.OrdersPath(id))
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, o := range orders {
		w.WriteString(strings.Join(o, " "))
		w.WriteByte('\n')
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func (s Store) ReadOrders(id string) ([][]string, error) {
	f, err := os.Open(s.OrdersPath(id))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	var out [][]string
	for sc.Scan() {
		if line := sc.Text(); line != "" {
			out = append(out, strings.Fields(line))
		}
	}
	return out, sc.Err()
}
