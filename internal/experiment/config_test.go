package experiment

import (
	"os"
	"path/filepath"
	"testing"
)

func valid() Config {
	return Config{
		Name: "t",
		Seed: 1,
		Graphs: Graphs{
			Train: 2, Val: 1, Test: 2,
			Size:         IntRange{Min: 16, Max: 64},
			EntriesRatio: FloatRange{Min: 0.02, Max: 0.05},
			MinPrereq:    1,
			MaxPrereq:    IntRange{Min: 2, Max: 4},
			Decay:        FloatRange{Min: 0.7, Max: 0.9},
			Symbols:      Symbols{Charset: "ABC123", Length: 3},
		},
		Orders:             Orders{Markov: 10, AE: 5, MarkovBudgets: []int{2, 10}},
		Windows:            Windows{MinLength: 8, MaxLength: 32},
		ChallengesPerLevel: 2,
		AEWindows:          AEWindows{TrainPerGraph: 3, ValPerGraph: 1},
		AE:                 AE{Dim: 32, GNNLayers: 2, Layers: 2, Heads: 4, FFN: 64, RandomFeatures: 8, Batch: 16, Steps: 50, LR: 0.001, Warmup: 5, EvalEvery: 10},
	}
}

func TestValidConfigPasses(t *testing.T) {
	if err := valid().Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(*Config){
		"no test graphs":            func(c *Config) { c.Graphs.Test = 0 },
		"graph shorter than window": func(c *Config) { c.Graphs.Size.Min = 7 },
		"size range reversed":       func(c *Config) { c.Graphs.Size.Min, c.Graphs.Size.Max = 64, 16 },
		"entries ratio of one":      func(c *Config) { c.Graphs.EntriesRatio.Max = 1 },
		"no prerequisites":          func(c *Config) { c.Graphs.MinPrereq = 0 },
		"max below min prereq":      func(c *Config) { c.Graphs.MaxPrereq.Min = 0 },
		"decay above one":           func(c *Config) { c.Graphs.Decay.Max = 1.5 },
		"no markov orders":          func(c *Config) { c.Orders.Markov = 0 },
		"no ae orders":              func(c *Config) { c.Orders.AE = 0 },
		"window below 8":            func(c *Config) { c.Windows.MinLength = 7 },
		"window range reversed":     func(c *Config) { c.Windows.MaxLength = 7 },
		"no challenges":             func(c *Config) { c.ChallengesPerLevel = 0 },
		"reserved character":        func(c *Config) { c.Graphs.Symbols.Charset = "AB_" },
		"repeated character":        func(c *Config) { c.Graphs.Symbols.Charset = "AAB" },
		"space in charset":          func(c *Config) { c.Graphs.Symbols.Charset = "A B" },
		"too few symbols":           func(c *Config) { c.Graphs.Symbols.Length = 2 }, // 6^2 = 36 < 64
		"no budgets":                func(c *Config) { c.Orders.MarkovBudgets = nil },
		"budget above markov":       func(c *Config) { c.Orders.MarkovBudgets = []int{2, 11} },
		"budgets not increasing":    func(c *Config) { c.Orders.MarkovBudgets = []int{10, 2} },
		"zero budget":               func(c *Config) { c.Orders.MarkovBudgets = []int{0, 10} },
		"no val graphs for ae":      func(c *Config) { c.Graphs.Val = 0 },
		"dim not split by heads":    func(c *Config) { c.AE.Heads = 5 },
		"no training steps":         func(c *Config) { c.AE.Steps = 0 },
		"warmup past steps":         func(c *Config) { c.AE.Warmup = 50 },
		"no learning rate":          func(c *Config) { c.AE.LR = 0 },
	}
	for name, mutate := range cases {
		c := valid()
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: Validate returned no error", name)
		}
	}
}

func TestLoadRepoConfigs(t *testing.T) {
	for _, name := range []string{"base", "smoke"} {
		if _, err := Load(filepath.Join("..", "..", "experiments", name+".json")); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func TestLoadRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.json")
	if err := os.WriteFile(path, []byte(`{"name": "x", "sede": 1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("typo in a field name was accepted")
	}
}
