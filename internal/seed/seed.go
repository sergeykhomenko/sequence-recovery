// Package seed derives independent, reproducible random streams from one
// experiment seed, so results do not depend on processing order.
package seed

import (
	"encoding/binary"
	"hash/fnv"
	"math/rand/v2"
)

// Derive hashes the base seed together with the labels of a stream.
func Derive(base uint64, parts ...string) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], base)
	h.Write(buf[:])
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return h.Sum64()
}

// RNG returns the generator of the stream named by parts.
func RNG(base uint64, parts ...string) *rand.Rand {
	s := Derive(base, parts...)
	return rand.New(rand.NewPCG(s, s^0x9e3779b97f4a7c15))
}
