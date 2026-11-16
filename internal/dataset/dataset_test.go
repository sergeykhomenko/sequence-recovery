package dataset

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/sergeykhomenko/sequence-recovery/internal/damage"
	"github.com/sergeykhomenko/sequence-recovery/internal/experiment"
	"github.com/sergeykhomenko/sequence-recovery/internal/jsonl"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

func load(t *testing.T, name string) experiment.Config {
	t.Helper()
	c, err := experiment.Load(filepath.Join("..", "..", "experiments", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func build(t *testing.T) (experiment.Config, store.Store) {
	t.Helper()
	c := load(t, "smoke")
	st := store.Store{Dir: t.TempDir()}
	if err := Build(c, st); err != nil {
		t.Fatal(err)
	}
	return c, st
}

func TestPlanDrawsFloatingGraphs(t *testing.T) {
	plan := Plan(load(t, "base"))
	counts := map[string]int{}
	ids := map[string]bool{}
	var sizes []int
	for _, p := range plan {
		counts[p.Split]++
		ids[p.ID] = true
		sizes = append(sizes, p.Size)
		if p.Size < 16 || p.Size > 4096 || p.Entries < 1 || p.Entries >= p.Size ||
			p.MaxPrereq < 2 || p.MaxPrereq > 8 || p.Decay < 0.7 || p.Decay > 0.9 {
			t.Errorf("%s: parameters out of range: %+v", p.ID, p.Config)
		}
	}
	if counts[store.Train] != 180 || counts[store.Val] != 20 || counts[store.Test] != 40 {
		t.Errorf("splits = %v, want 180/20/40", counts)
	}
	if len(ids) != len(plan) {
		t.Error("graph IDs repeat")
	}
	slices.Sort(sizes)
	// Log-uniform on [16, 4096] has its median near √(16·4096) = 256; a uniform draw would be near 2056.
	if med := sizes[len(sizes)/2]; med < 128 || med > 512 {
		t.Errorf("median size %d, want about 256", med)
	}
}

func TestBuildIsDeterministic(t *testing.T) {
	c := load(t, "smoke")
	a, b := t.TempDir(), t.TempDir()
	for _, dir := range []string{a, b} {
		if err := Build(c, store.Store{Dir: dir}); err != nil {
			t.Fatal(err)
		}
	}
	files := 0
	err := filepath.WalkDir(a, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		files++
		rel, _ := filepath.Rel(a, path)
		x, _ := os.ReadFile(path)
		y, err := os.ReadFile(filepath.Join(b, rel))
		if err != nil || !bytes.Equal(x, y) {
			t.Errorf("%s differs between builds", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if graphs := c.Graphs.Train + c.Graphs.Val + c.Graphs.Test; files < 2*graphs {
		t.Errorf("only %d files written", files)
	}
}

func TestBuildProtectsExistingExperiment(t *testing.T) {
	for _, change := range []string{"same", "seed", "budgets", "missing config", "invalid config"} {
		t.Run(change, func(t *testing.T) {
			c, st := build(t)
			prediction := st.PredictionPath("sentinel")
			if err := os.WriteFile(prediction, []byte("existing predictions\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "seed":
				c.Seed++
			case "budgets":
				c.Orders.MarkovBudgets = c.Orders.MarkovBudgets[:1]
			case "missing config":
				if err := os.Remove(st.ConfigPath()); err != nil {
					t.Fatal(err)
				}
			case "invalid config":
				if err := os.WriteFile(st.ConfigPath(), []byte("{"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			before := snapshot(t, st.Dir)
			err := Build(c, st)
			if change == "same" && err != nil {
				t.Fatal(err)
			}
			if change != "same" && (err == nil || !strings.Contains(err.Error(), "new output directory")) {
				t.Fatalf("expected a new-directory error, got %v", err)
			}
			after := snapshot(t, st.Dir)
			if len(before) != len(after) {
				t.Fatal("output file count changed")
			}
			for name, data := range before {
				if !bytes.Equal(data, after[name]) {
					t.Errorf("existing file changed: %s", name)
				}
			}
		})
	}
}

func snapshot(t *testing.T, root string) map[string][]byte {
	t.Helper()
	files := map[string][]byte{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		files[path] = data
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func TestBuildWritesValidChallenges(t *testing.T) {
	c, st := build(t)
	challenges, err := jsonl.Read[task.Challenge](st.ChallengesPath())
	if err != nil {
		t.Fatal(err)
	}
	answers, err := jsonl.Read[task.Answer](st.AnswersPath())
	if err != nil {
		t.Fatal(err)
	}
	if want := c.Graphs.Test * len(damage.Levels) * c.ChallengesPerLevel; len(challenges) != want || len(answers) != want {
		t.Fatalf("%d challenges and %d answers, want %d", len(challenges), len(answers), want)
	}
	for i, ch := range challenges {
		a := answers[i]
		g, params, err := st.ReadGraph(ch.Graph)
		if err != nil {
			t.Fatal(err)
		}
		if a.ID != ch.ID || params.Split != store.Test {
			t.Fatalf("%s: answer %s, graph split %s", ch.ID, a.ID, params.Split)
		}
		if L := len(a.Tokens); L < c.Windows.MinLength || L > min(c.Windows.MaxLength, params.Size) || len(ch.Window) != L {
			t.Fatalf("%s: window of %d, answer of %d", ch.ID, len(ch.Window), L)
		}
		if len(ch.Window.MaskPositions()) == 0 {
			t.Fatalf("%s: no masks", ch.ID)
		}
		for j, tok := range ch.Window {
			if tok != "" && tok != a.Tokens[j] {
				t.Fatalf("%s: kept token %d differs from the answer", ch.ID, j)
			}
		}
		pos := map[string]int{}
		for j, tok := range a.Tokens {
			pos[tok] = j
		}
		for _, e := range g.Edges {
			from, ok1 := pos[e.From]
			to, ok2 := pos[e.To]
			if ok1 && ok2 && from > to {
				t.Fatalf("%s: %s stands after its dependant %s", ch.ID, e.From, e.To)
			}
		}
	}
}

func TestBuildWritesOrdersPerSplit(t *testing.T) {
	c, st := build(t)
	want := map[string]int{store.Train: c.Orders.AE, store.Val: c.Orders.AE, store.Test: c.Orders.Markov}
	for split, count := range want {
		ids, err := st.GraphIDs(split)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range ids {
			g, _, err := st.ReadGraph(id)
			if err != nil {
				t.Fatal(err)
			}
			orders, err := st.ReadOrders(id)
			if err != nil {
				t.Fatal(err)
			}
			if len(orders) != count {
				t.Errorf("%s (%s): %d orders, want %d", id, split, len(orders), count)
			}
			for _, o := range orders {
				if len(o) != len(g.Nodes) {
					t.Fatalf("%s: order of %d tokens for %d nodes", id, len(o), len(g.Nodes))
				}
			}
		}
	}
}

func TestBuildNamesNodesWithRandomSymbols(t *testing.T) {
	c, st := build(t)
	ids, err := st.GraphIDs(store.Test)
	if err != nil {
		t.Fatal(err)
	}
	g, _, err := st.ReadGraph(ids[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range g.Nodes {
		if len(n.ID) != c.Graphs.Symbols.Length || strings.Trim(n.ID, c.Graphs.Symbols.Charset) != "" {
			t.Errorf("node %q is not a random symbol", n.ID)
		}
	}
}

func TestBuildWritesTrainWindowsInsideOrders(t *testing.T) {
	c, st := build(t)
	for path, want := range map[string]struct {
		split string
		per   int
	}{
		st.AETrainPath(): {store.Train, c.AEWindows.TrainPerGraph},
		st.AEValPath():   {store.Val, c.AEWindows.ValPerGraph},
	} {
		windows, err := jsonl.Read[task.TrainWindow](path)
		if err != nil {
			t.Fatal(err)
		}
		ids, err := st.GraphIDs(want.split)
		if err != nil {
			t.Fatal(err)
		}
		if len(windows) != len(ids)*want.per {
			t.Errorf("%s: %d windows, want %d", path, len(windows), len(ids)*want.per)
		}
		for _, w := range windows {
			_, params, err := st.ReadGraph(w.Graph)
			if err != nil {
				t.Fatal(err)
			}
			orders, err := st.ReadOrders(w.Graph)
			if err != nil {
				t.Fatal(err)
			}
			if params.Split != want.split || w.Order >= len(orders) || w.Start+w.Length > len(orders[w.Order]) {
				t.Fatalf("%s: window %+v does not fit graph %s (%s)", path, w, w.Graph, params.Split)
			}
			for _, m := range w.Masked {
				if m >= w.Length {
					t.Fatalf("%s: mask %d outside a window of %d", path, m, w.Length)
				}
			}
		}
	}
}
