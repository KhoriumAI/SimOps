# Documentation Template: [Feature/Schema/Addition Name]

## 1. Overview
**Date:** [YYYY-MM-DD]
**Status:** [Draft / In Progress / Completed]
**Author:** [Your Name/Handle]

Brief description of the change. What problem does it solve or what new capability does it provide?

## 2. Motivation
Explain why this change is necessary. For example:
- "The current meshing algorithm fails on complex fillets."
- "Users need a way to export results to PDF."
- "The database schema needs to support multi-material properties."

## 3. Technical Implementation
Detail the changes made. Break this down into logical components.

### 3.1. Schema Changes (if applicable)
If you've modified a database schema, API response, or configuration format, document it here.

**Updated Fields:**
- `field_name`: [Type] - Description of the field and its valid values.
- `new_parameter`: [Type] - Explanation of the new parameter.

**Example JSON/Schema:**
```json
{
  "new_field": "value",
  "nested_object": {
    "key": "details"
  }
}
```

### 3.2. Logic & Algorithm Changes
Explain any new functions, classes, or logic flows introduced.

- `ModuleName.FunctionName()`: Describe the new logic.
- `NewComponent`: Describe its responsibility.

### 3.3. Dependencies
List any new packages or external tools required.

## 4. Usage / Integration
How can others use this new feature or schema?

**CLI Command:**
```bash
python scripts/my_new_feature.py --input data.stl
```

**API Usage:**
```python
result = process_data(input_params)
```

## 5. Testing & Validation
Describe how the changes were verified.

- **Unit Tests:** `tests/test_new_feature.py`
- **Validation Geometries:** List any CAD files or data used for testing.
- **Performance:** Impact on execution time or memory (if relevant).

## 6. Screenshots / Visuals (if applicable)
Include links to any visual evidence of the feature or fix.

## 7. Future Work / Known Limitations
- What is left to do?
- Any known bugs or edge cases not yet handled?


