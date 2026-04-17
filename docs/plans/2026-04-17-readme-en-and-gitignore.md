# README EN And Gitignore Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shorter English repository README, introduce a repository-level `.gitignore`, and remove tracked cache/archive noise without breaking the source template layout.

**Architecture:** Reuse the root Chinese README as the source of truth, but write `README.en.md` as a concise English mirror for GitHub visitors. Add `.gitignore` rules for system files, Python caches, zip archives, and generated runtime artifacts while explicitly preserving `auto-profiling/.auto-profiling/`, which is part of the checked-in runtime template.

**Tech Stack:** Markdown, `.gitignore`, git.

---

### Task 1: Write the English README

**Files:**
- Read: `/Users/tc/Downloads/inference-agent/README.md`
- Create: `/Users/tc/Downloads/inference-agent/README.en.md`

**Step 1: Draft the English README**

Cover:
- project overview
- core principles
- three scenarios
- quick start
- module boundaries
- verification commands

**Step 2: Review for GitHub readability**

Ensure links are relative paths and the document is shorter than the Chinese root README while preserving the same primary usage path.

### Task 2: Add `.gitignore`

**Files:**
- Create: `/Users/tc/Downloads/inference-agent/.gitignore`

**Step 1: Add ignore rules**

Ignore:
- `.DS_Store`
- `__pycache__/`
- `*.pyc`
- `*.zip`
- generated `.auto-profiling/` runtime artifacts

Explicitly keep:
- `auto-profiling/.auto-profiling/`

**Step 2: Verify rule intent**

Check that source templates remain tracked while generated runtime state outside that template path is ignored.

### Task 3: Remove tracked noise

**Files:**
- Remove tracked cache and archive files from the repository index and filesystem as appropriate

**Step 1: Remove noise**

Clean:
- `.DS_Store`
- tracked `__pycache__` directories
- tracked `*.zip`

**Step 2: Verify repository state**

Run: `git status --short`
Expected: only README, `.gitignore`, plan, and cleanup removals appear.

### Task 4: Publish the update

**Files:**
- Commit and push repository changes

**Step 1: Commit**

Run: `git add README.en.md .gitignore docs/plans/2026-04-17-readme-en-and-gitignore.md ... && git commit -m "docs: add English README and clean repo artifacts"`
Expected: commit succeeds.

**Step 2: Push**

Run: `git push origin main`
Expected: GitHub reflects the new English README and cleaner repository contents.
