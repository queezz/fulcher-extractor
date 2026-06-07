# AGENTS_COMMIT.md

Use readable commit messages that are easy to scan in Cursor, VS Code, and
Git history.

## Format

- Start with a short imperative title, ideally under 60 characters.
- Keep body lines narrow, around 72 characters when practical.
- Use a short bullet list when it clarifies what changed.
- Keep the body focused on user-visible or workflow-relevant changes.
- Add this trailer at the end:

```text
agent: codex
```

## Example

```text
Add line-fit QC PDF page

- Compose all fitted line groups for one frame into a PDF page
- Keep individual line PNG output opt-in
- Add synthetic smoke coverage and a manual check notebook

agent: codex
```
