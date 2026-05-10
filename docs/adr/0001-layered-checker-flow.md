# ADR 0001: Keep Probes, Checks, and Rendering Separate

## Status

Accepted

## Context

This project needs to inspect a live Linux system, reason about DaVinci Resolve compatibility, and present the results in both human-readable and machine-readable forms.

Those concerns change for different reasons:

- probe code changes when system tools or parsing details change
- compatibility rules change when GPU, driver, or package expectations change
- output code changes when the CLI or JSON contract changes

If these concerns are mixed together, tests become harder to write and behavior changes become harder to review safely.

## Decision

The codebase keeps three distinct layers:

- `probes/` for host inspection and parsing
- `checks/` for pure compatibility decisions over `SystemState`
- `render.py` for text and JSON presentation

Shared typed contracts live in `models.py`, and the CLI only orchestrates the end-to-end flow.

## Consequences

### Positive

- checks are easy to test with fixtures and mocks
- output format changes do not require probe changes
- new compatibility rules can be added without shelling out from the rules layer
- the same results can be rendered for humans or automation

### Trade-offs

- new data needs explicit plumbing through `SystemState`
- some changes touch multiple files when a new fact is probed and then validated

## Follow-up Guidance

When adding new behavior, first decide whether it belongs to probing, compatibility logic, shared models, or rendering. Avoid collapsing those layers just to save a small amount of code.
