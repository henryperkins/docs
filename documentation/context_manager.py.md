# Table of Contents

1. [Overview](#overview)
2. [Code Structure](#code-structure)
3. [Dependencies](#dependencies)
4. [Metrics](#metrics)

---

# Overview

**File:** `context_manager.py`  
**Language:** python  
**Path:** `context_manager.py`  

## Summary

This file contains the definition of the ContextManager class, which is responsible for managing persistent context information in AI interactions. It provides methods to add, retrieve, clear, remove, and query context entries.

## Recent Changes




# Code Structure

## Classes

### ContextManager

Manages persistent context information for AI interactions. This includes adding new context entries, retrieving all current entries, clearing entries, and querying for relevant entries based on a provided query.

#### Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `__init__` | Initializes the ContextManager with a maximum number of entries it can hold. | 1 |
| `add_context` | Adds a new context entry to the context manager. | 1 |
| `get_context` | Retrieves all current context entries. | 1 |
| `clear_context` | Clears all context entries. | 1 |
| `remove_context` | Removes context entries that contain the specified reference. | 3 |
| `get_relevant_context` | Retrieves the most relevant context entries based on the query. | 3 |


# Dependencies

```mermaid
graph TD;
```

# Metrics

## Code Quality

| Metric | Value | Status |
|--------|-------|--------|
| Maintainability | 70.2 | ⚠️ Warning |
## Complexity Metrics

| Metric | Value |
|--------|--------|
| Volume | 15.5 |
| Difficulty | 1.0 |
| Effort | 15.5 |