# Root README And Publish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a root `README.md` for external GitHub users that explains what `inference-agent` is, how to use it, and how the three optimization modules relate to `auto-profiling`, then publish the update.

**Architecture:** Create one repository-level README that acts as the homepage and router. Reuse existing module docs for detail, but keep the root document focused on orientation, quick start, scenario selection, module boundaries, and verification commands. Verify the README content and repository state, then commit and push the documentation update.

**Tech Stack:** Markdown, git, GitHub CLI, existing project docs.

---

### Task 1: Plan the README structure

**Files:**
- Read: `/Users/tc/Downloads/inference-agent/auto-profiling/README.md`
- Read: `/Users/tc/Downloads/inference-agent/llm-serving-opt-skill/SKILL.md`
- Create: `/Users/tc/Downloads/inference-agent/docs/plans/2026-04-17-root-readme-and-publish.md`

**Step 1: Define the sections**

Use an external-user-first structure:
- project overview
- what problems it solves
- repository layout
- quick start
- scenario selection
- module summaries
- verification commands
- GitHub usage notes

**Step 2: Verify the structure is sufficient**

Check that a first-time visitor could understand:
- what the repo is for
- where to start
- which file to edit
- which command to run next

### Task 2: Write the root README

**Files:**
- Create: `/Users/tc/Downloads/inference-agent/README.md`

**Step 1: Write the README**

Include:
- concise intro
- exactness-first philosophy
- three-module plus `auto-profiling` architecture
- end-to-end quick start commands
- recommended workflow by scenario
- test and validation commands

**Step 2: Review for clarity**

Check that the README is useful without reading internal planning files first.

### Task 3: Verify repository state

**Files:**
- Verify modified files

**Step 1: Check git status**

Run: `git status --short`
Expected: only the new README and plan file appear as changes.

**Step 2: Optionally inspect README content**

Run: `sed -n '1,240p' README.md`
Expected: the document includes overview, quick start, scenarios, and validation sections.

### Task 4: Publish the update

**Files:**
- Commit and push repository changes

**Step 1: Commit**

Run: `git add README.md docs/plans/2026-04-17-root-readme-and-publish.md && git commit -m "docs: add root repository README"`
Expected: commit succeeds.

**Step 2: Push**

Run: `git push origin main`
Expected: GitHub repository updates with the new README.
