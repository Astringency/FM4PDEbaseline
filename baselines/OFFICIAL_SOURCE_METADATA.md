# Official Source Metadata

Reviewer-facing result JSON/CSV rows include backend provenance fields from
`baselines/methods/official.py`. The vendored source snapshots under `offical/`
do not currently carry verified upstream commit/tag records, so
`official_commit_or_version` and `official_local_modifications` remain
`unknown` until manually audited.

To update a record, identify the exact upstream commit/tag used to populate the
vendored directory, compare the vendored tree against that upstream revision,
then update `OFFICIAL_SOURCE_INFO` in `baselines/methods/official.py`.

| source_key | official_repo | vendored_path | official_commit_or_version | official_local_modifications | how_to_update_this_record |
| --- | --- | --- | --- | --- | --- |
| neuraloperator | https://github.com/neuraloperator/neuraloperator | `offical/neuraloperator` | unknown | unknown | Record the upstream neuraloperator commit/tag and diff this vendored tree against it. |
| deepxde | https://github.com/lululxvi/deepxde | `offical/deepxde` | unknown | unknown | Record the upstream DeepXDE commit/tag and diff this vendored tree against it. |
| recfno | https://github.com/zhaoxiaoyu1995/recfno | `offical/RecFNO` | unknown | unknown | Record the upstream RecFNO commit/tag and diff this vendored tree against it. |
| senseiver | https://github.com/OrchardLANL/Senseiver | `offical/Senseiver` | unknown | unknown | Record the upstream Senseiver commit/tag and diff this vendored tree against it. |
| pc_bnn | https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning | `offical/PC-BNN` | unknown | unknown | Record the upstream PC-BNN commit/tag and diff this vendored tree against it. |
| ifno | https://github.com/BayesianAIGroup/iFNO | `offical/iFNO` | unknown | unknown | Record the upstream iFNO commit/tag and diff this vendored tree against it. |
| vivid | https://github.com/DL-WG/VIVID | `offical/VIVID` | unknown | unknown | Record the upstream VIVID commit/tag and diff this vendored tree against it. |
| invobs | https://github.com/googleinterns/invobs-data-assimilation | `offical/invobs-data-assimilation` | unknown | unknown | Record the upstream invobs commit/tag and diff this vendored tree against it. |
| vivid_invobs | https://github.com/DL-WG/VIVID; https://github.com/googleinterns/invobs-data-assimilation | `offical/VIVID`; `offical/invobs-data-assimilation` | unknown | unknown | Update both VIVID and invobs records, then keep this combined adapter record in sync. |
| voronoi_cnn | https://github.com/kfukami/Voronoi-CNN | `offical/Voronoi-CNN` | unknown | unknown | Record the upstream Voronoi-CNN commit/tag and diff this vendored tree against it. |

Official-aligned or official-architecture reimplementations must remain labeled
as reimplementations in result metadata; they are not direct official code.

The current Burgers Var4D/VIVID implementations are more distant local
adaptations and are not official-aligned reimplementations. Their implementation
metadata and official/native eligibility are defined in
`baselines/methods/var4d.py`, `baselines/methods/vivid.py`, and
`baselines/capabilities.py`.
