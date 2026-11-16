// Package experiment holds the configuration of one experiment run.
package experiment

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"

	"github.com/sergeykhomenko/sequence-recovery/internal/damage"
)

type IntRange struct {
	Min int `json:"min"`
	Max int `json:"max"`
}

type FloatRange struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

type Symbols struct {
	Charset string `json:"charset"`
	Length  int    `json:"length"`
}

type Graphs struct {
	Train        int        `json:"train"`
	Val          int        `json:"val"`
	Test         int        `json:"test"`
	Size         IntRange   `json:"size"`
	EntriesRatio FloatRange `json:"entries_ratio"`
	MinPrereq    int        `json:"min_prereq"`
	MaxPrereq    IntRange   `json:"max_prereq"`
	Decay        FloatRange `json:"decay"`
	Symbols      Symbols    `json:"symbols"`
}

type Orders struct {
	Markov        int   `json:"markov"`
	AE            int   `json:"ae"`
	MarkovBudgets []int `json:"markov_budgets"` // Markov is trained on the first K of the Markov trajectories
}

// AE holds the autoencoder hyperparameters, read by the Python trainer.
type AE struct {
	Dim            int     `json:"dim"`
	GNNLayers      int     `json:"gnn_layers"`
	Layers         int     `json:"layers"`
	Heads          int     `json:"heads"`
	FFN            int     `json:"ffn"`
	RandomFeatures int     `json:"random_features"`
	Batch          int     `json:"batch"`
	Steps          int     `json:"steps"`
	LR             float64 `json:"lr"`
	Warmup         int     `json:"warmup"`
	EvalEvery      int     `json:"eval_every"`
}

type Windows struct {
	MinLength int `json:"min_length"`
	MaxLength int `json:"max_length"`
}

type AEWindows struct {
	TrainPerGraph int `json:"train_per_graph"`
	ValPerGraph   int `json:"val_per_graph"`
}

type Config struct {
	Name               string    `json:"name"`
	Seed               uint64    `json:"seed"`
	Graphs             Graphs    `json:"graphs"`
	Orders             Orders    `json:"orders"`
	Windows            Windows   `json:"windows"`
	ChallengesPerLevel int       `json:"challenges_per_level"`
	AEWindows          AEWindows `json:"ae_windows"`
	AE                 AE        `json:"ae"`
}

// Load reads a config, rejecting unknown fields, and validates it.
func Load(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var c Config
	if err := dec.Decode(&c); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	if err := c.Validate(); err != nil {
		return Config{}, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

func (c Config) Validate() error {
	g := c.Graphs
	switch {
	case g.Test < 1:
		return errors.New("graphs.test must be >= 1")
	case g.Train < 0 || g.Val < 0:
		return errors.New("graphs.train and graphs.val must be >= 0")
	case g.Size.Min < c.Windows.MinLength || g.Size.Min > g.Size.Max:
		return errors.New("graphs.size must satisfy windows.min_length <= min <= max")
	case g.EntriesRatio.Min <= 0 || g.EntriesRatio.Min > g.EntriesRatio.Max || g.EntriesRatio.Max >= 1:
		return errors.New("graphs.entries_ratio must satisfy 0 < min <= max < 1")
	case g.MinPrereq < 1 || g.MaxPrereq.Min < g.MinPrereq || g.MaxPrereq.Min > g.MaxPrereq.Max:
		return errors.New("graphs.max_prereq must satisfy 1 <= min_prereq <= min <= max")
	case g.Decay.Min <= 0 || g.Decay.Min > g.Decay.Max || g.Decay.Max > 1:
		return errors.New("graphs.decay must satisfy 0 < min <= max <= 1")
	case c.Orders.Markov < 1:
		return errors.New("orders.markov must be >= 1")
	case g.Train+g.Val > 0 && c.Orders.AE < 1:
		return errors.New("orders.ae must be >= 1 when there are train or val graphs")
	case c.Windows.MinLength < damage.MinLength || c.Windows.MinLength > c.Windows.MaxLength:
		return fmt.Errorf("windows must satisfy %d <= min_length <= max_length", damage.MinLength)
	case c.ChallengesPerLevel < 1:
		return errors.New("challenges_per_level must be >= 1")
	case c.AEWindows.TrainPerGraph < 0 || c.AEWindows.ValPerGraph < 0:
		return errors.New("ae_windows counts must be >= 0")
	}
	if err := validateBudgets(c.Orders); err != nil {
		return err
	}
	if g.Train > 0 {
		if err := validateAE(c.AE, g.Val); err != nil {
			return err
		}
	}
	return validateSymbols(g.Symbols, g.Size.Max)
}

func validateBudgets(o Orders) error {
	b := o.MarkovBudgets
	if len(b) == 0 {
		return errors.New("orders.markov_budgets must list at least one budget")
	}
	for i, k := range b {
		if k < 1 || k > o.Markov || (i > 0 && k <= b[i-1]) {
			return fmt.Errorf("orders.markov_budgets must increase within 1..orders.markov (%d)", o.Markov)
		}
	}
	return nil
}

func validateAE(a AE, valGraphs int) error {
	switch {
	case valGraphs < 1:
		return errors.New("the autoencoder needs graphs.val >= 1 to select its model")
	case a.Dim < 1 || a.GNNLayers < 1 || a.Layers < 1 || a.Heads < 1 || a.FFN < 1 ||
		a.RandomFeatures < 1 || a.Batch < 1 || a.Steps < 1 || a.EvalEvery < 1:
		return errors.New("ae sizes, batch, steps and eval_every must be >= 1")
	case a.Dim%a.Heads != 0:
		return errors.New("ae.dim must be divisible by ae.heads")
	case a.LR <= 0:
		return errors.New("ae.lr must be > 0")
	case a.Warmup < 0 || a.Warmup >= a.Steps:
		return errors.New("ae.warmup must satisfy 0 <= warmup < steps")
	}
	return nil
}

func validateSymbols(s Symbols, maxSize int) error {
	if s.Length < 1 || len(s.Charset) < 2 {
		return errors.New("graphs.symbols needs length >= 1 and at least 2 characters")
	}
	seen := map[rune]bool{}
	for _, r := range s.Charset {
		switch {
		case r > unicode.MaxASCII || !unicode.IsPrint(r) || unicode.IsSpace(r):
			return fmt.Errorf("graphs.symbols.charset: %q is not a printable ASCII character", r)
		case strings.ContainsRune("^$_", r):
			return fmt.Errorf("graphs.symbols.charset: %q is reserved by gomarkov", r)
		case seen[r]:
			return fmt.Errorf("graphs.symbols.charset: %q repeats", r)
		}
		seen[r] = true
	}
	if math.Pow(float64(len(s.Charset)), float64(s.Length)) < float64(maxSize) {
		return errors.New("graphs.symbols cannot name graphs.size.max distinct nodes")
	}
	return nil
}
