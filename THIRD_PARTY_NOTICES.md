# Third-party attribution

FinNews uses **GPT Researcher** by Assaf Elovic and its contributors as an installed
dependency (`gpt-researcher==0.17.0`): https://github.com/assafelovic/gpt-researcher.
GPT Researcher is MIT licensed. Its copyright and license remain in the distributed
upstream package. FinNews does not claim authorship of its planning, collection or
report-writing engine, and does not vendor or modify its source.

The proposal referenced upstream commit
`ddd4db36208d9c8e49f3b2436efda97baecb5433`; implementation targets the released 0.17.0
package after checking `conduct_research()`, `get_research_sources()` and
`write_report(ext_context=..., custom_prompt=...)`.

Other principal components: Sentence Transformers (Apache-2.0),
`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0 model card), rank-bm25
(Apache-2.0), NumPy (BSD-3-Clause), Pydantic (MIT), Streamlit (Apache-2.0),
tiktoken (MIT), HTTPX (BSD-3-Clause). Consult installed distributions/model cards
for authoritative license texts and dependency notices.

The `data/demo` text is self-authored fictional test data under this repository's
MIT license. Financial company names identify the intended application domain;
the synthetic numerical statements do not describe those companies. Official
third-party pages retain their respective owners' rights. The repository publishes
source links and download tooling, not a redistribution of full news corpora.
