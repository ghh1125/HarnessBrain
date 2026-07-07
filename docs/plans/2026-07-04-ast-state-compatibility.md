# AST State Compatibility

## Goal

Replace keyword-count-only state compatibility with an optional component-level
AST structural similarity signal, while preserving the existing evolution,
evidence, and steering flow.

## Design

1. Encode each modified component as a JSON-serializable normalized AST
   extracted from the candidate harness. The representation keeps Python
   syntax structure but ignores identifier and literal values.
2. When a new harness is evaluated, compare its component AST with the most
   recent historical AST associated with each evidence family using ordered
   tree edit distance with unit insertion, deletion, and substitution costs.
3. Store the resulting similarity in the existing `context_discount` field, so
   downstream maturity and guidance logic remains unchanged.
4. If source code, component functions, or valid ASTs are unavailable, retain
   the existing keyword-based transition discount as a compatibility fallback.

The ordered tree edit distance is computed with an internal Zhang--Shasha
dynamic program. The implementation uses only the Python standard library and
introduces no new runtime dependency. Existing flattened signatures are treated
as unavailable and follow the established compatibility fallback.

## Verification

- Structurally identical functions with renamed identifiers have tree edit
  distance `0` and similarity `1.0`.
- Inserting one AST node has tree edit distance `1`.
- Structural changes reduce normalized similarity according to
  `1 - TED / (|T_e| + |T_t|)`.
- Evidence encoding stores component AST signatures when candidate source is
  available.
- Transition migration uses AST similarity when both signatures exist and
  falls back to the existing behavior otherwise.
