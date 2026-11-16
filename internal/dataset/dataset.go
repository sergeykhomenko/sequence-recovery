// Package dataset builds an experiment: graphs with floating size and random
// symbols, their trajectories, test challenges and autoencoder windows.
package dataset

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"reflect"
	"strconv"

	"github.com/sergeykhomenko/sequence-recovery/internal/alphabet"
	"github.com/sergeykhomenko/sequence-recovery/internal/damage"
	"github.com/sergeykhomenko/sequence-recovery/internal/experiment"
	"github.com/sergeykhomenko/sequence-recovery/internal/generator"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
	"github.com/sergeykhomenko/sequence-recovery/internal/jsonl"
	"github.com/sergeykhomenko/sequence-recovery/internal/sampler"
	"github.com/sergeykhomenko/sequence-recovery/internal/seed"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

// Plan draws the parameters of every graph. IDs run g0001, g0002, ... over
// the train, val and test splits in that order.
func Plan(c experiment.Config) []store.GraphParams {
	g := c.Graphs
	var out []store.GraphParams
	for _, split := range []struct {
		name  string
		count int
	}{{store.Train, g.Train}, {store.Val, g.Val}, {store.Test, g.Test}} {
		for range split.count {
			id := fmt.Sprintf("g%04d", len(out)+1)
			rng := seed.RNG(c.Seed, "plan", id)
			size := logUniform(rng, g.Size.Min, g.Size.Max)
			ratio := uniform(rng, g.EntriesRatio.Min, g.EntriesRatio.Max)
			out = append(out, store.GraphParams{
				ID:    id,
				Split: split.name,
				Config: generator.Config{
					Size:      size,
					Entries:   max(1, int(math.Round(ratio*float64(size)))),
					MinPrereq: g.MinPrereq,
					MaxPrereq: g.MaxPrereq.Min + rng.IntN(g.MaxPrereq.Max-g.MaxPrereq.Min+1),
					Decay:     uniform(rng, g.Decay.Min, g.Decay.Max),
					Seed:      seed.Derive(c.Seed, "graph", id),
				},
			})
		}
	}
	return out
}

// Build writes the whole experiment into st.
func Build(c experiment.Config, st store.Store) error {
	if err := checkOutput(c, st); err != nil {
		return err
	}
	if err := st.Init(); err != nil {
		return err
	}
	if err := st.WriteConfig(c); err != nil {
		return err
	}

	var challenges []task.Challenge
	var answers []task.Answer
	var aeTrain, aeVal []task.TrainWindow
	for _, p := range Plan(c) {
		g, err := makeGraph(c, p)
		if err != nil {
			return fmt.Errorf("%s: %w", p.ID, err)
		}
		if err := st.WriteGraph(g, p); err != nil {
			return err
		}
		s, err := sampler.New(g)
		if err != nil {
			return fmt.Errorf("%s: %w", p.ID, err)
		}

		count := c.Orders.AE
		if p.Split == store.Test {
			count = c.Orders.Markov
		}
		rng := seed.RNG(c.Seed, "orders", p.ID)
		orders := make([][]string, count)
		for i := range orders {
			orders[i] = s.Sample(rng)
		}
		if err := st.WriteOrders(p.ID, orders); err != nil {
			return err
		}

		switch p.Split {
		case store.Test:
			cs, as, err := makeChallenges(c, p, s)
			if err != nil {
				return fmt.Errorf("%s: %w", p.ID, err)
			}
			challenges = append(challenges, cs...)
			answers = append(answers, as...)
		case store.Train:
			ws, err := makeTrainWindows(c, p, orders, c.AEWindows.TrainPerGraph)
			if err != nil {
				return fmt.Errorf("%s: %w", p.ID, err)
			}
			aeTrain = append(aeTrain, ws...)
		case store.Val:
			ws, err := makeTrainWindows(c, p, orders, c.AEWindows.ValPerGraph)
			if err != nil {
				return fmt.Errorf("%s: %w", p.ID, err)
			}
			aeVal = append(aeVal, ws...)
		}
	}

	if err := jsonl.Write(st.ChallengesPath(), challenges); err != nil {
		return err
	}
	if err := jsonl.Write(st.AnswersPath(), answers); err != nil {
		return err
	}
	if err := jsonl.Write(st.AETrainPath(), aeTrain); err != nil {
		return err
	}
	return jsonl.Write(st.AEValPath(), aeVal)
}

// checkOutput rejects incompatible or unidentified output before any file is changed.
// Rebuilding an existing experiment with the same configuration is deterministic.
func checkOutput(c experiment.Config, st store.Store) error {
	entries, err := os.ReadDir(st.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(entries) == 0 {
		return nil
	}
	previous, err := st.ReadConfig()
	if err != nil {
		return fmt.Errorf("%s: cannot verify the existing experiment configuration; use a new output directory: %w", st.Dir, err)
	}
	if !reflect.DeepEqual(previous, c) {
		return fmt.Errorf("%s: already contains a different experiment configuration; use a new output directory", st.Dir)
	}
	return nil
}

// makeGraph generates the graph and renames its nodes to random symbols.
func makeGraph(c experiment.Config, p store.GraphParams) (*graph.Graph, error) {
	g, err := generator.Generate(p.Config, generator.DecayStrategy{Decay: p.Decay})
	if err != nil {
		return nil, err
	}
	rng := seed.RNG(c.Seed, "symbols", p.ID)
	names, err := alphabet.Random(rng, len(g.Nodes), c.Graphs.Symbols.Charset, c.Graphs.Symbols.Length)
	if err != nil {
		return nil, err
	}
	mapping := make(map[string]string, len(names))
	for i, n := range g.Nodes {
		mapping[n.ID] = names[i]
	}
	out, err := g.Relabel(mapping)
	if err != nil {
		return nil, err
	}
	out.Shuffle(rng)
	return out, nil
}

// makeChallenges cuts every challenge from its own fresh trajectory, which is
// never written anywhere.
func makeChallenges(c experiment.Config, p store.GraphParams, s *sampler.Sampler) ([]task.Challenge, []task.Answer, error) {
	var cs []task.Challenge
	var as []task.Answer
	for _, level := range damage.Levels {
		for k := range c.ChallengesPerLevel {
			id := fmt.Sprintf("%s-%s-%03d", p.ID, level, k)
			rng := seed.RNG(c.Seed, "challenge", id)
			order := s.Sample(rng)
			start, length := window(rng, len(order), c.Windows)
			tokens := order[start : start+length]
			masked, err := damage.Mask(rng, level, length)
			if err != nil {
				return nil, nil, err
			}
			cs = append(cs, task.Challenge{ID: id, Graph: p.ID, Level: string(level), Window: task.NewWindow(tokens, masked)})
			as = append(as, task.Answer{ID: id, Tokens: tokens})
		}
	}
	return cs, as, nil
}

// makeTrainWindows draws autoencoder windows from the graph's stored trajectories.
func makeTrainWindows(c experiment.Config, p store.GraphParams, orders [][]string, count int) ([]task.TrainWindow, error) {
	out := make([]task.TrainWindow, 0, count)
	for k := range count {
		rng := seed.RNG(c.Seed, "ae", p.ID, strconv.Itoa(k))
		order := rng.IntN(len(orders))
		start, length := window(rng, len(orders[order]), c.Windows)
		level := damage.Levels[rng.IntN(len(damage.Levels))]
		masked, err := damage.Mask(rng, level, length)
		if err != nil {
			return nil, err
		}
		out = append(out, task.TrainWindow{Graph: p.ID, Order: order, Start: start, Length: length, Level: string(level), Masked: masked})
	}
	return out, nil
}

// window draws the start and length of a window in a trajectory of n tokens.
func window(rng *rand.Rand, n int, w experiment.Windows) (start, length int) {
	length = w.MinLength + rng.IntN(min(w.MaxLength, n)-w.MinLength+1)
	return rng.IntN(n - length + 1), length
}

func logUniform(rng *rand.Rand, lo, hi int) int {
	x := math.Exp(uniform(rng, math.Log(float64(lo)), math.Log(float64(hi))))
	return min(hi, max(lo, int(math.Round(x))))
}

func uniform(rng *rand.Rand, lo, hi float64) float64 {
	return lo + rng.Float64()*(hi-lo)
}
