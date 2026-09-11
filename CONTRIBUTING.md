# Contributing

Thanks for your interest in Nova. It's a deliberately small codebase, so
contributions are easy to review.

## Setup

```bash
git clone https://github.com/escape9th/nova-agent.git
cd nova-agent
pip install -e ".[dev]"
```

## Making a change

1. Open an issue describing the change (or comment on an existing one).
2. Create a branch and implement the change.
3. Add or update tests in `tests/`.
4. Run the full suite and make sure it passes:

   ```bash
   pytest
   ```

5. Keep commits focused; use conventional prefixes (`feat:`, `fix:`, `docs:`,
   `test:`, `refactor:`, `chore:`).

## Style

- Target Python 3.10+.
- Type hints on public functions; docstrings on non-obvious ones.
- Match the surrounding code's style — Nova favours small, explicit functions.

## Scope

Nova aims to stay minimal. If a feature would substantially grow the core,
consider whether it belongs in an example or an extension first.
