package markov

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Model is a read-only view of a chain written by Trainer.Export: for every
// context n-gram, how often each token followed it.
type Model struct {
	Order int
	next  map[string]map[string]int
}

// chainFile is gomarkov's JSON layout.
type chainFile struct {
	Order    int                       `json:"int"`
	SpoolMap map[string]int            `json:"spool_map"`
	FreqMat  map[string]map[string]int `json:"freq_mat"`
}

func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f chainFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	names := make(map[int]string, len(f.SpoolMap))
	for s, i := range f.SpoolMap {
		names[i] = s
	}
	m := &Model{Order: f.Order, next: make(map[string]map[string]int, len(f.FreqMat))}
	for from, row := range f.FreqMat {
		ctx, err := state(names, from)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		counts := make(map[string]int, len(row))
		for to, n := range row {
			tok, err := state(names, to)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			counts[tok] = n
		}
		m.next[ctx] = counts
	}
	return m, nil
}

func state(names map[int]string, key string) (string, error) {
	i, err := strconv.Atoi(key)
	if err != nil {
		return "", err
	}
	s, ok := names[i]
	if !ok {
		return "", fmt.Errorf("state %d is not in spool_map", i)
	}
	return s, nil
}

// Next returns how often each token followed context (len(context) == Order),
// or nil when the context never occurred.
func (m *Model) Next(context []string) map[string]int {
	return m.next[strings.Join(context, "_")]
}
