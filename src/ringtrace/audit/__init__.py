"""Signal audits: does a row group, a feature set or a template carry information at all?

Each audit answers a yes/no question cheaply, before a full cascade run is spent on it:

* ``probes``: label-free row groups from the device counter (ordinary fraud-pass rows against
  legit-pass rows, in train and in October alike);
* ``stats``: cross-validated AUC, per-feature tests, entity-link shares and an adversarial
  comparison of two periods;
* ``ceiling``: the label-free composition of a month and the macro-F1 a perfect caller of the
  reachable rows would score;
* ``lookalike``: whether the sibling type inside a look-alike template can be told apart;
* ``shuffle``: an order probe of a submission (same content, rows permuted).

Entry point: ``python -m ringtrace.audit <command>``. Outputs are aggregates only.
"""
