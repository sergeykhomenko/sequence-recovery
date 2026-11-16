// Package task defines what the recovery methods exchange: challenges
// (windows with masks), answers, predictions and autoencoder training windows.
package task

import "encoding/json"

// Window is a fragment of a trajectory; "" marks a mask and is written as JSON null.
type Window []string

func NewWindow(tokens []string, masked []int) Window {
	w := make(Window, len(tokens))
	copy(w, tokens)
	for _, p := range masked {
		w[p] = ""
	}
	return w
}

func (w Window) MaskPositions() []int {
	var pos []int
	for i, t := range w {
		if t == "" {
			pos = append(pos, i)
		}
	}
	return pos
}

func (w Window) MarshalJSON() ([]byte, error) {
	out := make([]*string, len(w))
	for i := range w {
		if w[i] != "" {
			out[i] = &w[i]
		}
	}
	return json.Marshal(out)
}

func (w *Window) UnmarshalJSON(b []byte) error {
	var in []*string
	if err := json.Unmarshal(b, &in); err != nil {
		return err
	}
	*w = make(Window, len(in))
	for i, t := range in {
		if t != nil {
			(*w)[i] = *t
		}
	}
	return nil
}

type Challenge struct {
	ID     string `json:"id"`
	Graph  string `json:"graph"`
	Level  string `json:"level"`
	Window Window `json:"window"`
}

type Answer struct {
	ID     string   `json:"id"`
	Tokens []string `json:"tokens"`
}

// Prediction fills the masks of a challenge left to right. OrdersUsed is set
// by Markov: the order each token came from, 0 for a random fallback.
type Prediction struct {
	ID         string   `json:"id"`
	Fill       []string `json:"fill"`
	OrdersUsed []int    `json:"orders_used,omitempty"`
}

// TrainWindow points at a window of orders/<Graph>.txt for the autoencoder.
type TrainWindow struct {
	Graph  string `json:"graph"`
	Order  int    `json:"order"`
	Start  int    `json:"start"`
	Length int    `json:"length"`
	Level  string `json:"level"`
	Masked []int  `json:"masked"`
}
