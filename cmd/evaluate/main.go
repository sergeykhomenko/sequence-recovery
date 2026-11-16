// Command evaluate scores every prediction file of an experiment against the
// answers and writes metrics.csv, summary.csv and comparisons.csv.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sergeykhomenko/sequence-recovery/internal/damage"
	"github.com/sergeykhomenko/sequence-recovery/internal/graph"
	"github.com/sergeykhomenko/sequence-recovery/internal/jsonl"
	"github.com/sergeykhomenko/sequence-recovery/internal/metrics"
	"github.com/sergeykhomenko/sequence-recovery/internal/report"
	"github.com/sergeykhomenko/sequence-recovery/internal/store"
	"github.com/sergeykhomenko/sequence-recovery/internal/task"
)

func main() {
	dir := flag.String("experiment", "", "experiment directory")
	reference := flag.String("reference", "ae", "method compared with all others in comparisons.csv")
	iters := flag.Int("bootstrap", 1000, "bootstrap resamples per confidence interval")
	flag.Parse()
	if *dir == "" {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(store.Store{Dir: *dir}, *reference, *iters); err != nil {
		log.Fatal(err)
	}
}

type graphInfo struct {
	size    int
	closure *graph.Closure
}

func run(st store.Store, reference string, iters int) error {
	cfg, err := st.ReadConfig()
	if err != nil {
		return err
	}
	challenges, err := jsonl.Read[task.Challenge](st.ChallengesPath())
	if err != nil {
		return err
	}
	answerList, err := jsonl.Read[task.Answer](st.AnswersPath())
	if err != nil {
		return err
	}
	answers := make(map[string][]string, len(answerList))
	for _, a := range answerList {
		answers[a.ID] = a.Tokens
	}
	graphs := map[string]graphInfo{}
	info := func(id string) (graphInfo, error) {
		if gi, ok := graphs[id]; ok {
			return gi, nil
		}
		g, _, err := st.ReadGraph(id)
		if err != nil {
			return graphInfo{}, err
		}
		cl, err := graph.NewClosure(g)
		if err != nil {
			return graphInfo{}, fmt.Errorf("%s: %w", id, err)
		}
		graphs[id] = graphInfo{size: len(g.Nodes), closure: cl}
		return graphs[id], nil
	}

	paths, err := filepath.Glob(filepath.Join(st.PredictionsDir(), "*.jsonl"))
	if err != nil {
		return err
	}
	sort.Strings(paths)
	var rows []report.Row
	for _, path := range paths {
		method := strings.TrimSuffix(filepath.Base(path), ".jsonl")
		preds, err := jsonl.Read[task.Prediction](path)
		if err != nil {
			return err
		}
		byID := make(map[string]task.Prediction, len(preds))
		for _, p := range preds {
			byID[p.ID] = p
		}
		for _, c := range challenges {
			p, ok := byID[c.ID]
			if !ok {
				return fmt.Errorf("%s has no prediction for %s", method, c.ID)
			}
			gi, err := info(c.Graph)
			if err != nil {
				return err
			}
			res, err := metrics.Score(answers[c.ID], c.Window, p.Fill, gi.closure)
			if err != nil {
				return fmt.Errorf("%s, %s: %w", method, c.ID, err)
			}
			masked := c.Window.MaskPositions()
			order, budget := report.ParseMethod(method)
			row := report.Row{
				Method: method, Order: order, Budget: budget, Challenge: c.ID, Graph: c.Graph,
				Size: gi.size, Length: len(c.Window), Level: c.Level,
				Ratio: float64(len(masked)) / float64(len(c.Window)), Gaps: damage.Gaps(masked),
				EPA: res.EPA, OAR: res.OAR, UAR: res.UAR, PVR: res.PVR,
			}
			if res.Valid {
				row.VWR = 1
			}
			if row.Order > 0 {
				row.Backoff, row.Fallback = report.Backoff(p.OrdersUsed, row.Order)
			}
			rows = append(rows, row)
		}
		fmt.Printf("scored %s\n", method)
	}
	if len(rows) == 0 {
		return fmt.Errorf("%s: no predictions", st.PredictionsDir())
	}

	if err := report.WriteRows(st.MetricsPath(), rows); err != nil {
		return err
	}
	if err := report.WriteSummary(st.SummaryPath(), report.Summarize(rows, iters, cfg.Seed)); err != nil {
		return err
	}
	comparisons, err := report.Compare(rows, reference)
	if err != nil {
		return err
	}
	if len(comparisons) == 0 {
		fmt.Printf("no predictions for reference %q, comparisons.csv has only a header\n", reference)
	}
	return report.WriteComparisons(st.ComparisonsPath(), comparisons)
}
