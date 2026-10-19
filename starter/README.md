# Starter kit

```
data/pdfs/            14 annual reports: the corpus your app answers from
data/golden_set.json  16 questions about them, with answers and evidence pages
eval.py               asks your app every question and scores the answers
```

## The golden set

Each question looks like this:

```json
{
  "id": "q06",
  "question": "What was the value of Total power generation capacity (GW) of RWE AG ...",
  "kind": "number",
  "answerable": true,
  "answer": "39.3",
  "documents": ["RWE AG.pdf"],
  "evidence": [[{"document": "RWE AG.pdf", "page": 47}]]
}
```

- `kind` is `number`, `boolean`, `name` or `names`.
- `answerable: false` means the report doesn't contain the answer, so the right
  response is to say so. There are 3 of these.
- `evidence` lists the pages the answer is on, numbered from 1 as in a PDF
  viewer. Each inner list is a group of pages, and any one page in the group is
  enough. Most questions have one group. The comparison question (q10) has one
  group per report it needs.
- Numbers are in full units: `406100000`, not `406.1` (million).

## Running the eval

```bash
python eval.py --label agent                  # needs your API running on :8000
python eval.py --label pipeline --api http://localhost:8001
```

Edit `ask()` at the top of `eval.py` to call your API. It has to return the
answer text and the pages it cites, as the PDF's file name and a 1-based page
number. Don't edit the scoring below it.

Each run writes `runs/<label>.json` and prints the scorecard:

| metric | what it means |
|---|---|
| answer accuracy | correct answers to the answerable questions |
| correct refusals on unanswerable | "not in the report" when it isn't |
| evidence cited | the citations include a page the answer is on (±1 page) |
| seconds per question | wall-clock time, measured by the eval |
| usd for the whole set | what `ask()` reports as cost |

The scoring rules are simple on purpose. A number counts if any number in the
answer is within 1% of the expected one ("$406.1 million" matches `406100000`).
Yes/no is read from the first yes/no/true/false in the answer. A name counts if
it appears in the answer. Read `is_correct()` and you'll know exactly how you
are graded.
